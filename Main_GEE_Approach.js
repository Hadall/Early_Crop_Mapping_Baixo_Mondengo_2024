var balanced = projects/ee-adalbertodissertation/assets/balanced_points
var ROI = projects/ee-adalbertodissertation/assets/ROI
var initial_time = 'yyyy-mm-dd';
var final_time = 'yyyy-mm-dd';
var cloud_cover_thrshold = 10;
var sentinel_1_bands = ['VV', 'VH']; 
var sentinel_2_bands = ['B1', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12']; 


function calculate_ndvi(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

function reprojectTo10m(image) {
  return image
    .reproject({crs: 'EPSG:32629', scale: 10})
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    });
}

function calculate_ndwi(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return image.addBands(ndwi);
}

function reduce_to_10_days_mean(collection_name, start_date, end_date, interval_days) {
    var ee_start_date = ee.Date(start_date);
    var ee_end_date = ee.Date(end_date);
    var total_days = ee_end_date.difference(ee_start_date, 'day');
    var sequence = ee.List.sequence(0, total_days, interval_days);

    var medians_of_collection = sequence.map(function(i) {
        var start_of_stage = ee_start_date.advance(i, 'day');
        var end_of_stage = start_of_stage.advance(interval_days, 'day');
        var median_of_stage = collection_name.filterDate(start_of_stage, end_of_stage)
                                             .median()
                                             .set('system:time_start', start_of_stage.millis());
        return reprojectTo10m(median_of_stage);
    });
    return medians_of_collection;
}

function add_features_date_to_name(image, medians_of_collection) {
    var new_band_names = medians_of_collection.map(function(img) {
        var date = ee.Image(img).date().format('YYYY-MM-dd');
        return ee.Image(img).bandNames().map(function(bandName) {
            return ee.String(bandName).cat('_').cat(date);
        });
    }).flatten();

    return image.rename(new_band_names);
}

function Z_Score_to_Image(image, bands, region) {
    var mean = image.select(bands).reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region,
        scale: 10,
        maxPixels: 1e9
    });
    var stdDev = image.select(bands).reduceRegion({
        reducer: ee.Reducer.stdDev(),
        geometry: region,
        scale: 10,
        maxPixels: 1e9
    });
    return image.select(bands).subtract(mean.toImage(bands)).divide(stdDev.toImage(bands));
}

function calculate_stats(collection, bands) {
    var means = {};
    var stdDevs = {};
  
    bands.forEach(function(band) {
      var mean = collection.reduceColumns(ee.Reducer.mean(), [band]).get('mean');
      var stdDev = collection.reduceColumns(ee.Reducer.stdDev(), [band]).get('stdDev');
      means[band] = mean;
      stdDevs[band] = stdDev;
    });
  
    return {means: means, stdDevs: stdDevs};
  }

  function Z_Score_to_Bands(feature, bands, stats) {
    var normDict = {};
    bands.forEach(function(band) {
      var value = ee.Number(feature.get(band));
      var mean = ee.Number(stats.means[band]);
      var stdDev = ee.Number(stats.stdDevs[band]);
      var normalizedValue = value.subtract(mean).divide(stdDev);
      normDict[band] = normalizedValue;
    });
    return feature.set(normDict);
  }
  
  function Z_Score_to_Collections(feature) {
    feature = Z_Score_to_Bands(feature, sentinel1Bands, S_1_Stats); //Cooment this line in case of approach B
    feature = Z_Score_to_Bands(feature, sentinel2Bands, S_2_Stats);
    return feature;
  }

var filtered_sentinel_2_imgs = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(ROI)
    .filterDate(ee.Date(initial_time), ee.Date(final_time))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_thrshold))
    .select(sentinel_2_bands);
    
var filtered_sentinel_2_imgs_and_SI = filtered_sentinel_2_imgs.map(calculate_ndvi).map(calculate_ndwi);

var filtered_sentinel_1_imgs = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
    .filterBounds(ROI)
    .filterDate(ee.Date(initial_time), ee.Date(final_time))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(sentinel_1_bands);

var filtered_sentinel_2_imgs_resampled = filtered_sentinel_2_imgs_and_SI.map(reprojectTo10m);
var filtered_sentinel_1_imgs_resampled = filtered_sentinel_1_imgs.map(reprojectTo10m);
var s1_images = filtered_sentinel_1_imgs_resampled;
var s2_images = filtered_sentinel_2_imgs_resampled;
var reduced_s1_images = reduce_to_10_days_mean(filtered_sentinel_1_imgs_resampled, initial_time, final_time, 10);
var reduced_s2_images = reduce_to_10_days_mean(filtered_sentinel_2_imgs_resampled, initial_time, final_time, 10);
var s1_bands = add_features_date_to_name(ee.ImageCollection.fromImages(reduced_s1_images).toBands(), reduced_s1_images);
var s2_bands = add_features_date_to_name(ee.ImageCollection.fromImages(reduced_s2_images).toBands(), reduced_s2_images);
var fused_satellite_data = s1_bands.addBands(s2_bands);
var s1_s2_data = fused_satellite_data.clip(ROI);

var multiespectral_band_indexis = ee.List.sequence('int_number','int_number'); 
var multiespctral_set_bands = s1_s2_data.select(multiespectral_band_indexis);

//RANDOM FOREST CASE
var sampleData = s1_s2_data.sampleRegions({ //Change s1_s2_data to multiespctral_set_band in case of Approach B
  collection: sample_points,
  properties: ['class_ID'], 
  scale: 10,
  geometries: true
});

var sampleData_ = sampleData.randomColumn('random')
var split_threshold = 0.7
var training = sampleData_.filter(ee.Filter.lt('random', split_threshold));
var testing = sampleData_.filter(ee.Filter.gte('random', split_threshold));
var balanced_training_set = balanced;
var best_features_for_RF = s1_s2_data.select(['Best Feature Names'])
var nomes_s1_s2_data = s1_s2_data.bandNames() 


var trainedClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 307,            
  variablesPerSplit: 3, 
  minLeafPopulation: 26, 
  bagFraction: 1.0,              
  maxNodes: 18,                  
  seed: 0                        
}).train({
  features: balanced,
  classProperty: 'class_ID',
  inputProperties: best_features_for_RF.bandNames()  
});
var crop_map =  s1_s2_data.classify(trainedClassifier); //Change s1_s2_data to multiespctral_set_band in case of Approach B


//SUPPORT VECTOR MACHINE CASE
var s1_s2_data_without_normalization = s1_s2_data
var s1_normalized_data = Z_Score_to_Image(s1_s2_data_without_normalization, ['S-1 Best Features'], ROI); 
var s2_normalized_data = Z_Score_to_Image(s1_s2_data_without_normalization, ['S-2 Best Features'], ROI); 
var s1_s2_normalized_data = s1_normalized_data.addBands(s2_normalized_data);
var spectral_index = s1_s2_data_without_normalization.select(['SIs']) //In order to maintain the index scales
var s1_s2_IS_normalized = s1_s2_normalized_data.addBands(spectral_index); 

var only_s2_and_IS = s2_normalized_data.addBands(spectral_index); //In Case of Aproach B

var sampleData = s1_s2_data_without_normalization.sampleRegions({ //to avoide data leakage..., 
    //...sampling is not performed on the normalized multiband image
  collection: sample_points,
  properties: ['class_ID'], 
  scale: 10,
  geometries: true
});
var sampleData_ = sampleData.randomColumn('random')
var split_threshold = 0.7
var training = sampleData_.filter(ee.Filter.lt('random', split_threshold));
var testing = sampleData_.filter(ee.Filter.gte('random', split_threshold));
var balanced_training_set = balanced;

var S_1_Bands = ['S-1 Best Features'];  
var S_2_Bands = ['S-2 Best Features',];  
var indices = ['SI Best Features']; 
var S_1_Stats = calculate_stats(balanced_training_set, S_1_Bands);
var S_2_Stats = calculate_stats(balanced_training_set, S_2_Bands);
var normalized_training = balanced_training_set.map(Z_Score_to_Collections);


var trainedClassifier = ee.Classifier.libsvm({
  kernelType: 'linear',
  cost: 9.6
}).train({
  features: normalized_training,
  classProperty: 'class_ID',
  inputProperties: s1_s2_IS_normalized.bandNames() //change to only_s2_and_IS.bandNames() in case of approach B
});

var crop_map =  s1_s2_IS_normalized.classify(trainedClassifier); //change to only_s2_and_IS in case of approach B


var classes = [1, 2, 3, 4, 5, 6];
var testingSample = crop_map.sampleRegions({
  collection: testing,
  properties: ['class_ID'],
  scale: 10
});

var error_matrix = testingSample.errorMatrix('class_ID', 'classification', classes);
print('Confusion Matrix', error_matrix);
print('Accuracy', error_matrix.accuracy());
var producerAccuracy = error_matrix.producersAccuracy();
var consumerAccuracy = error_matrix.consumersAccuracy(); 
print('Producer Accuracy (Recall)', producerAccuracy);
print('Consumer Accuracy (Precision)', consumerAccuracy);
var F1_score = error_matrix.fscore()
print('F1 Score', F1_score)
var f1ScoreGeneral = ee.Array(F1_score).reduce(ee.Reducer.mean(), [0]).get([0]);
print('General F1 Score', f1ScoreGeneral);


// var rgb_view = s1_s2_data.visualize({
//   bands: ['B4', 'B3', 'B2'], 
//   min: 0, 
//   max: 3000, 
//   gamma: [0.95, 0.95, 0.95] 
// });
// Map.addLayer(rgb_view, {}, 'RGB Data View')

//Plot Classified Map

Map.addLayer(crop_map, {
  min: 1,
  max: 6,
  palette: ['yellow', 'green', 'blue', 'grey','red', 'cyan','green']
}, 'Crop Map');


Export.image.toDrive({
  image: crop_map,
  description: 'crop_map', 
  scale: 10, 
  region: ROI, 
  maxPixels: 1e13,
  crs:'EPSG:3763'
});


