# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:58:19 2021

@author: morenodu
"""


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# define dataset
df_cli2_scaled = df_clim_avg_features
df_t_scaled = df_epic_det
df_severe =pd.DataFrame( np.where(df_t_scaled < df_t_scaled.mean()-df_t_scaled.std(),True, False), index = df_t_scaled.index,columns = ['severe_loss'] ).astype(int)
loss_intensity = df_severe
X, y = df_cli2_scaled, loss_intensity
#divide data train and test
X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, loss_intensity, test_size=0.3, random_state=0)

# evaluate the model
model_cb = CatBoostClassifier(verbose=0, n_estimators=100 )
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_cb, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# fit the model on the whole dataset
model = CatBoostClassifier(verbose=0, n_estimators=100)
model.fit(X_train, y_train)
# make a single prediction
# row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
# yhat = model.predict(row)
# print('Prediction: %d' % yhat[0])

shap_values = model.get_feature_importance(Pool(X_train, y_train), type='ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

# visualize the first prediction's explanation
shap.force_plot(expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True)

shap.force_plot(expected_value, shap_values, X_train,matplotlib=True)

test_objects = [X_train.iloc[6:7], X_train.iloc[7:8]]

for obj in test_objects:
    print('Probability of class 1 = {:.4f}'.format(model.predict_proba(obj)[0][1]))
    print('Formula raw prediction = {:.4f}'.format(model.predict(obj, prediction_type='RawFormulaVal')[0]))
    print('\n')
shap.force_plot(expected_value, shap_values[6,:], X.iloc[6,:], matplotlib=True)

shap.summary_plot(shap_values, X_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)


shap.force_plot(explainer.expected_value, shap_values, X,matplotlib=True)

shap.summary_plot(shap_values, X_train)

shap.initjs() 
shap_values_2012 = explainer.shap_values( X_train.iloc[[6]])
shap_display = shap.force_plot(explainer.expected_value[1], shap_values_2012[1], X_train.iloc[[6]],matplotlib=True)

display(shap_display)