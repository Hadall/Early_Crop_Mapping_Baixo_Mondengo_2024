import pandas as pd
from imblearn.over_sampling import SMOTE
import geopandas as gpd
from shapely.geometry import Point
import json

umbalanced_training_set = pd.read_csv('/content/drive/MyDrive/GEE_Exports/imbalanced_data.csv')

umbalanced_training_set['geometry'] = umbalanced_training_set['.geo'].apply(lambda x: Point(json.loads(x)['coordinates']))
umbalanced_training_set['LAT'] = umbalanced_training_set['geometry'].apply(lambda g: g.y)
umbalanced_training_set['LON'] = umbalanced_training_set['geometry'].apply(lambda g: g.x)

X = umbalanced_training_set.drop(['class_ID', '.geo', 'geometry', 'random', 'system:index'], axis=1)
y = umbalanced_training_set['class_ID']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

balanced_training_set = pd.DataFrame(X_resampled, columns=X.columns)
balanced_training_set['class_ID'] = y_resampled
balanced_training_set['lat'] = umbalanced_training_set.iloc[:len(y_resampled)]['LAT']
balanced_training_set['lon'] = umbalanced_training_set.iloc[:len(y_resampled)]['LON']

balanced_training_set['lat'] = pd.to_numeric(balanced_training_set['LAT'], errors='coerce')
balanced_training_set['lon'] = pd.to_numeric(balanced_training_set['LON'], errors='coerce')

balanced_training_set.to_csv('/content/drive/MyDrive/Dados/balanced_data.csv', index=False)