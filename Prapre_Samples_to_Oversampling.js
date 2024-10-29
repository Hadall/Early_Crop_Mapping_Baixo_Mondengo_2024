var imbalaced_points = rojects/ee-adalbertodissertation/assets/imbalaced_points

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

function reproject_to_10m(image) {
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
        return reproject_to_10m(median_of_stage);
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

var filtered_sentinel_2_imgs_resampled = filtered_sentinel_2_imgs_and_SI.map(reprojectTo10m);
var filtered_sentinel_1_imgs_resampled = filtered_sentinel_1_imgs.map(reprojectTo10m);
var s1_images = filtered_sentinel_1_imgs_resampled;
var s2_images = filtered_sentinel_2_imgs_resampled;
var reduced_s1_images = reduce_to_10_days_mean(filtered_sentinel_1_imgs_resampled, initial_time, final_time, 10);
var reduced_s2_images = reduce_to_10_days_mean(filtered_sentinel_2_imgs_resampled, initial_time, final_time, 10);
var s1_bands = add_features_date_to_name(ee.ImageCollection.fromImages(reduced_s1_images).toBands(), reduced_s1_images);
var s2_bands = add_features_date_to_name(ee.ImageCollection.fromImages(reduced_s2_images).toBands(), reduced_s2_images);
var fused_satellite_data = s1_bands.addBands(s2_bands);
var s1_s2_data = fused_satellite_data.clip(ROI)


var sampleData = s1_s2_data.sampleRegions({ 
    collection: sample_points,  
    properties: ['class_ID'], 
    scale: 10,
    geometries: true
  });
  var sampleData_ = sampleData.randomColumn('random')
  var split_threshold = 0.7
  var training = sampleData_.filter(ee.Filter.lt('random', split_threshold));
  var testing = sampleData_.filter(ee.Filter.gte('random', split_threshold));

  var number_of_sentinel_1_imgs = filtered_sentinel_1_imgs.size();
  var S1_list = filtered_sentinel_1_imgs.toList(filtered_sentinel_1_imgs.size());
  var S1_dates = S1_list.map(function(image) {
    return ee.Image(image).date().format();
  });

  var number_of_sentinel_2_imgs = filtered_sentinel_2_imgs_and_SI.size();
  var S2_list = filtered_sentinel_2_imgs.toList(filtered_sentinel_2_imgs.size());
  var S2_dates = S2_list.map(function(image) {
    return ee.Image(image).date().format();
  });

  Export.table.toDrive({
    collection: training,
    description: 'imbalanced_training_data',
    folder: 'GEE_Exports',
    fileNamePrefix: 'training_data',
    fileFormat: 'CSV'
  });
  