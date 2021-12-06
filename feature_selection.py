# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:00:42 2021

@author: morenodu
"""
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)

# The "accuracy" scoring is proportional to the number of correct
# classifications

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=model, step=1, scoring='r2',
              min_features_to_select=min_features_to_select)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()




df_clim_mon_brs_sub = df_clim_mon_brs.loc[:,['DTR1', 'DTR2', 'DTR3', 'ETR2','R10mm12',  'R10mm2', 'Rx1day12', 'Rx5day12']]

feature_importance_selection(df_clim_mon_brs_sub, df_obs_mean_det)

X, y = df_clim_mon_brs_sub.values, df_obs_mean_det.values.ravel()

y_pred_ece, y_pred_total_ece = calibration(X,y)
df_pred_ece = pd.DataFrame(y_pred_ece, index = df_obs_test.index)
df_pred_ece_total = pd.DataFrame(y_pred_total_ece, index = df_clim_mon_brs.index)
