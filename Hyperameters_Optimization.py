from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Dados/balanced_data_.csv')
clean_data = df.drop(columns= ['LAT','LON','lat','lon'])

#Feature Selection
#S1
sentinel_1_df = clean_data.loc[:, clean_data.columns.str.startswith('V')]
sentinel_1_df['class_ID'] = clean_data['class_ID']
X_s1 = sentinel_1_df.drop('class_ID', axis=1)
y_s1 = sentinel_1_df['class_ID']
Xs1_train, Xs1_test, ys1_train, ys1_test = train_test_split(X_s1, y_s1, test_size=0.30, random_state=42)

RF_base_model_s1 = RandomForestClassifier(random_state=42)
rfecv_RF_1 = RFECV(estimator=RF_base_model_s1, step=1, cv=5, scoring='accuracy')
rfecv_RF_1.fit(Xs1_train, ys1_train)

s1_feature_names = Xs1_train.columns
cv_scores_RF = rfecv_RF_1.cv_results_['mean_test_score']
s1_accuracy_changes = np.ediff1d(cv_scores_RF, to_begin=cv_scores_RF[0] - cv_scores_RF[0])  

s1_feature_names = Xs1_train.columns
s1_feature_importances = rfecv_RF_1.estimator_.feature_importances_
feature_data = pd.DataFrame({
    'Feature': s1_feature_names,
    'Importance': np.concatenate(([0] * (len(s1_feature_names) - len(s1_feature_importances)), s1_feature_importances)),
    'Selected': ['Yes' if x else 'No' for x in rfecv_RF_1.support_],
    'Accuracy Change': s1_accuracy_changes
})
s1_feature_data = feature_data[['Feature', 'Selected', 'Importance', 'Accuracy Change']]


#S2
sentinel_2_df = clean_data.loc[:, ~clean_data.columns.str.startswith('V')] 
sentinel_2_df['class_ID'] = clean_data['class_ID']
sentinel_2_df['class_ID'] = clean_data['class_ID']
X_s2 = sentinel_2_df.drop('class_ID', axis=1)
y_s2 = sentinel_2_df['class_ID']

RF_base_model_2 = RandomForestClassifier(random_state=42)
rfecv_RF_2 = RFECV(estimator=RF_base_model, step=1, cv=5, scoring='accuracy') 
rfecv_RF_2.fit(Xs2_train, ys2_train)


cv_scores_RF_2 = rfecv_RF_2.cv_results_['mean_test_score']
s2_accuracy_changes = np.ediff1d(cv_scores_RF_2, to_begin=cv_scores_RF_2[0] - cv_scores_RF_2[0]) 
s2_feature_names = Xs2_train.columns
s2_feature_importances = rfecv_RF_2.estimator_.feature_importances_
feature_data_2 = pd.DataFrame({
    'Feature': s2_feature_names,
    'Importance': np.concatenate(([0] * (len(s2_feature_names) - len(s2_feature_importances)), s2_feature_importances)),
    'Selected': ['Yes' if x else 'No' for x in rfecv_RF_2.support_],
    'Accuracy Change': s2_accuracy_changes
})
s2_feature_data_2 = feature_data_2[['Feature', 'Selected', 'Importance', 'Accuracy Change']]


s2_feature_data_2.to_csv('/content/drive/MyDrive/Dados/S1_feature_selection.csv')
s1_feature_data.to_csv('/content/drive/MyDrive/Dados/sS2_feature_selection.csv')


#RF HYPERPARAMTER OPTIMIZATION
X = clean_data.drop('class_ID', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

p_distributions_RF = {
    'n_estimators': randint(100, 500),
    'max_features': uniform(0.1, 0.9),
    'max_depth': randint(5, 50),
    'min_samples_split': uniform(0.01, 0.1),
    'min_samples_leaf': uniform(0.01, 0.1),
    'bootstrap': [True, False]
}

RF_classifier = RandomForestClassifier()

random_search_RF = RandomizedSearchCV(
    estimator=RF_classifier,
    param_distributions=p_distributions_RF,
    n_iter=10, 
    cv=5, 
    random_state=42,
    n_jobs=-1
)

random_search_RF.fit(X_train, y_train)

best_RF_model = random_search_RF.best_estimator_
best_RF_model_score = best_RF_model.score(X_test, y_test)
print("Performance of the best RF model on the test set:", best_RF_model_score)
best_RF_parameters = random_search_RF.best_params_
print('Best_Hyperparameter:', best_RF_parameters)

#Standardization Min Leaf Population and Variables Per Split for Entry into GEE
total_samples = len(X_train)
min_leaf_fraction = best_RF_parameters['min_samples_leaf']
min_leaf_population = int(round(total_samples * min_leaf_fraction))
print(min_leaf_population)
total_features = X_train.shape[1]
fration = best_RF_parameters['max_features']
variablesPerSplit = int(round(total_features * fration))
variablesPerSplit