import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/drive/MyDrive/Dados/balanced_data.csv')
clean_data = df.drop(columns= ['LAT','LON','lat','lon'])

#FEATURE SELECTION
sentinel_1_df = clean_data.loc[:, clean_data.columns.str.startswith('V')] #Because of all Sentinel-1 Features Starts with 'V'.                           
sentinel_1_df['class_ID'] = clean_data['class_ID']
X_s1 = sentinel_1_df.drop('class_ID', axis=1)
y_s1 = sentinel_1_df['class_ID']
Xs1_train, Xs1_test, ys1_train, ys1_test = train_test_split(X_s1, y_s1, test_size=0.30, random_state=42)

pipe_s1 = Pipeline([
    ('scaler', StandardScaler()),  
    ('svm', SVC(kernel='linear', random_state=42)) 
])
rfecv_SVM_for_s1 = RFECV(
    estimator=pipe_s1,
    step=1,
    cv=5,
    scoring='accuracy',
    importance_getter='named_steps.svm.coef_'  
)
rfecv_SVM_for_s1.fit(Xs1_train, ys1_train)


s1_feature_names = Xs1_train.columns
cv_scores_SVM_1 = rfecv_SVM_for_s1.cv_results_['mean_test_score']

kernel = pipe_s1.named_steps['svm'].kernel
if kernel == 'linear':
    svm_model = rfecv_SVM_for_s1.estimator_.named_steps['svm']
    coef_matrix_1 = svm_model.coef_
    importance_s1 = np.mean(np.abs(coef_matrix_1), axis=0)
    selected_features = Xs1_train.columns[rfecv_SVM_for_s1.support_]
    feature_importance = dict(zip(selected_features, importance_s1))
    feature_data_1 = pd.DataFrame({
        'Feature': Xs1_train.columns,
        'Importance': [feature_importance.get(x, 0) for x in Xs1_train.columns],
        'Selected': ['Yes' if x else 'No' for x in rfecv_SVM_for_s1.support_],
        'Accuracy Change': np.ediff1d(rfecv_SVM_for_s1.cv_results_['mean_test_score'], to_begin=0)
    })
else:
    feature_data_1 = pd.DataFrame({
        'Feature': Xs1_train.columns,
        'Importance': [0] * len(Xs1_train.columns),
        'Selected': ['Yes' if x else 'No' for x in rfecv_SVM_for_s1.support_],
        'Accuracy Change': np.ediff1d(rfecv_SVM_for_s1.cv_results_['mean_test_score'], to_begin=0)
    })


sentinel_2_df = clean_data.loc[:, ~clean_data.columns.str.startswith('V')]
sentinel_2_df['class_ID'] = clean_data['class_ID']
sentinel_2_df['class_ID'] = clean_data['class_ID']
X_s2 = sentinel_2_df.drop('class_ID', axis=1)
y_s2 = sentinel_2_df['class_ID']
Xs2_train, Xs2_test, ys2_train, ys2_test = train_test_split(X_s2, y_s2, test_size=0.30, random_state=42)

pipe_s2 = Pipeline([
    ('scaler', StandardScaler()),  
    ('svm', SVC(kernel='linear', random_state=42))  
])

rfecv_SVM_for_s2 = RFECV(
    estimator=pipe_s2,
    step=1,
    cv=5,
    scoring='accuracy',
    importance_getter='named_steps.svm.coef_'  
)
rfecv_SVM_for_s2.fit(Xs2_train, ys2_train)

s2_feature_names = Xs2_train.columns
cv_scores_SVM_2 = rfecv_SVM_for_s2.cv_results_['mean_test_score']

kernel_2 = pipe_s2.named_steps['svm'].kernel

if kernel == 'linear':
    svm_model_2 = rfecv_SVM_for_s2.estimator_.named_steps['svm']
    coef_matrix_2 = svm_model_2.coef_
    importance_s2 = np.mean(np.abs(coef_matrix_2), axis=0)
    selected_features = Xs2_train.columns[rfecv_SVM_for_s2.support_]
    feature_importance = dict(zip(selected_features, importance_s2))
    feature_data_2 = pd.DataFrame({
        'Feature': Xs2_train.columns,
        'Importance': [feature_importance.get(x, 0) for x in Xs2_train.columns],
        'Selected': ['Yes' if x else 'No' for x in rfecv_SVM_for_s2.support_],
        'Accuracy Change': np.ediff1d(rfecv_SVM_for_s2.cv_results_['mean_test_score'], to_begin=0)
    })
else:
    feature_data_2 = pd.DataFrame({
        'Feature': Xs2_train.columns,
        'Importance': [0] * len(Xs2_train.columns),
        'Selected': ['Yes' if x else 'No' for x in rfecv_SVM_for_s2.support_],
        'Accuracy Change': np.ediff1d(rfecv_SVM_for_s2.cv_results_['mean_test_score'], to_begin=0)
    })


feature_data_1.to_csv('/content/drive/MyDrive/Dados_SVM/s1_fetaure_selection.csv')
feature_data_2.to_csv('/content/drive/MyDrive/Dados_SVM/s2_feature_selection.csv')

#HYPERPARAMTERS OPTIMIZATION
scaler = StandardScaler()
X_s1_scalested = scaler.fit_transform(X_s1)
X_s1_scalested = pd.DataFrame(X_s1_scalested, columns=X_s1.columns)
X_s2_scalested = scaler.fit_transform(X_s2)
X_s2_scalested = pd.DataFrame(X_s2_scalested, columns=X_s2.columns)
s1_s2_df = pd.concat([X_s1_scalested, X_s2_scalested], axis=1)
s1_s2_df['class_ID'] = y_s1  

X_train_scalested, X_test_scalested, y_train_scalested, y_test_scalested = train_test_split(
    s1_s2_df.drop('class_ID', axis=1), s1_s2_df['class_ID'], test_size=0.30, random_state=42
)

svm_classifier = SVC(kernel='linear')
param_distributions = {
    'C': uniform(0.1, 10)  
}
random_search = RandomizedSearchCV(
    estimator=svm_classifier,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_scalested, y_train_scalested)
best_parameters = random_search.best_params_
print('Melhores hiperparâmetros:', best_parameters)