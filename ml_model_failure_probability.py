import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.ticker as plticker

from  scipy import signal 
from mask_shape_border import mask_shape_border
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression

#%% US mask
# Crop space to either US or soy states
usa = gpd.read_file('gadm36_USA_1.shp', crs="epsg:4326") 
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio', 'South Dakota','North Dakota', 'Missouri','Arkansas']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]

#%% yield model - WOFOST
DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = DS_y.reindex(lat=DS_y.lat[::-1])
DS_y = mask_shape_border(DS_y,soy_us_states ) #clipping for us
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')

df_wofost=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
da_wofost_std = DS_y['yield'].std('time')

# uncomment if you want to see the maps
# plt.figure(figsize=(20,10)) 
# ax=plt.axes(projection=ccrs.Mercator())
# DS_y['yield'].std('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='afmhot_r',levels=10)
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()
#%% climate features
# DS_t_mean=xr.open_dataset("cru/cru_tmp.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_t_max=xr.open_dataset("cru/cru_tmx.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
# DS_t_min=xr.open_dataset("cru/cru_tmn.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_prec=xr.open_dataset("cru/cru_pre.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_evap=xr.open_dataset("cru/cru_vap.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_wet=xr.open_dataset("cru/cru_wet.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_spei = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
# DS_spei3 = xr.open_dataset("spei03.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
# DS_spei3 = DS_spei3.rename_vars({'spei':'speithree'})
# DS_spei4 = xr.open_dataset("spei04.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
# DS_spei4 = DS_spei4.rename_vars({'spei':'speifour'})

# Merge
DS_cli = xr.merge([DS_t_max.tmx,DS_evap.vap, DS_spei.spei]).sel(time=slice('1980-01-01','2010-12-31'))
DS_cli_us = mask_shape_border(DS_cli, soy_us_states) #US-shape
DS_cli_us = DS_cli_us.dropna(dim = 'lon', how='all')
DS_cli_us = DS_cli_us.dropna(dim = 'lat', how='all')

# #%% open data for cluster 3 
# cluster = 3 
# DS_cluster3 = DS_cli_us.where(cluster_yield == unique_values[cluster])
# DS_y_cluster3  = DS_y.where(cluster_yield == unique_values[cluster])

# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_cluster3.std('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'})
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()

# DS_cli_det_us = DS_cli_us.where(DS_cli_us.tmx > -300, -40000 )
#%% functions detrending climate features based on the US mask & temperature mask (values are dependent on it)
def detrend_feature(dataarray_reference, dataarray_in, reference_value, NA_value, months_selected):
    dataarray_iso = dataarray_in.where(dataarray_reference > reference_value, NA_value)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=dataarray_iso.attrs) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin(months_selected)) 
    dataarray_det_mean = dataarray_iso_2.groupby('time').mean(...)
    df_detrended = dataarray_det_mean.to_series()
    return df_detrended
# function converting dataframe into matrix for machine learning variables
def matrix_conv(df, months_selected):
    mat_df = df.values.reshape(int(len(df)/len(months_selected)),len(months_selected)) # divide by each month
    return mat_df
#%% Detrending cliamte features (CROP MODEL already detrended)
# data
months_selected = [6, 7, 8, 9, 10]
df_y_f = df_wofost 
### improve code
mat_cli_2 = [] 
for var_name, values in DS_cli_us.items():
    df = detrend_feature(DS_cli_us.tmx, DS_cli_us[var_name],-300, -30000, months_selected)
    mat_test = matrix_conv(df, months_selected)
    mat_cli_2.append(mat_test)
mat_cli_2 = np.concatenate(mat_cli_2, axis=1)
df_cli2 = pd.DataFrame(mat_cli_2, index =df_y_f.index )

column_names=[]
for i in list(DS_cli_us.keys()):
    for j in range(6,11):
        column_names.append(i+str(j))
df_cli2.columns = column_names
#remove some variables if necessary
# df_cli2 = df_cli2.iloc[:,0:15]
#get together
df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)

#%% fit linear regressions and plot them compared with the scatter plots and the respective R2 score:
def scatter_plot(feature, target):
    linear_regressor = LinearRegression()
    linear_regressor.fit(df_cli2[feature].values.reshape(-1,1), df_y_f[target])
    Y_pred = linear_regressor.predict(df_cli2[feature].values.reshape(-1,1))
    score = format(linear_regressor.score(df_cli2[feature].values.reshape(-1,1), df_y_f[target]),'.3f')

    plt.figure(figsize=(10,6))
    plt.scatter(df_cli2[feature], df_y_f[target], c='black')
    plt.title(f"R2 (0-1) score is {score}", fontsize=20)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel("Yield ton/ha", fontsize=16)
    plt.axhline(np.mean(df_y_f[target]))
    plt.axhline(np.mean(df_y_f[target]) - np.std(df_y_f[target]),linestyle='--' )
    plt.plot(df_cli2[feature], Y_pred, color='red')
    return score

score_set=[]
for i in df_cli2.columns.values:
    score_i = scatter_plot(i,'yield')
    score_set.append(float(score_i))

sc_set=pd.DataFrame(index = list(df_cli2.columns),data = score_set, columns=['R2_score'])
print('The maximum score is', sc_set.max().values, ', corresponding to the feature:', sc_set.R2_score.idxmax())
print(sc_set.sort_values(by=['R2_score'], ascending=False))

#%% Regularize/standard data
#standardized
scaler=StandardScaler()
df_cli2_scaled = pd.DataFrame(scaler.fit_transform(df_cli2), columns = df_cli2.columns, index=df_cli2.index)
# # SPEI is already scaled, so it needs to be returned to its original form (could you confirm?)
# df_cli2_scaled[["spei6", "spei7", "spei8", "spei9", "spei10"]] = df_cli2[["spei6", "spei7", "spei8", "spei9", "spei10"]]
df_t_scaled = pd.DataFrame(scaler.fit_transform(df_y_f),columns = df_y_f.columns, index=df_y_f.index)
df_scaled = pd.concat([df_cli2_scaled,df_t_scaled], axis=1, sort=False)
# failures - check them and see what explains each failure.
df_failures = df_scaled.loc[df_scaled['yield'] <= -1]
df_cond = df_failures[(df_failures <= -0.9) | (df_failures >= 0.9)]
df_cond_nonscaled = df_cli2[df_scaled['yield'] <= -1]
df_30 = df_scaled[df_cli2['tmx8'] > 30 ]
df_30 = df_30[(df_30 <= -0.9) | (df_30 >= 0.9)]
df_cat = pd.DataFrame( np.where(df_t_scaled < -1,'Failure',np.where(df_t_scaled > 1,'High', 'Normal')), index = df_t_scaled.index,columns = ['yield_category'] )
df_total_cat = pd.concat([df_scaled,df_cat], axis=1, sort=False)

#%% heatmap with the correlation of each feature + yield
corrmat = df_scaled.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (16,13))
g = sns.heatmap(df_scaled[top_corr_features].corr(),annot=True, cmap="RdYlGn")
plt.title("Pearson's correlation")
# kendall Rank Correlation
corrkendall = df_scaled.corr(method='kendall')
plt.figure(figsize = (16,13))
g = sns.heatmap(corrkendall, xticklabels=corrkendall.columns.values,yticklabels=corrkendall.columns.values, cmap="RdYlGn",annot=True)
plt.title("Kendall rank correlation")
plt.show()

# Get redundant variables and rank them
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5, chosen_method='pearson'):
    au_corr = df.corr(method=chosen_method).abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
# select the best features according to the Pearson's correlation
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y.T)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()[::-1]
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

# Pearsons
print("Top Pearsons Correlations \n", get_top_abs_correlations(df_cli2_scaled, 10))
# Kendall
print("Top Kendalls Correlations \n", get_top_abs_correlations(df_cli2_scaled, 10, chosen_method = 'kendall'))
cor_support, cor_feature = cor_selector(df_cli2_scaled, df_t_scaled, 6)
print("\n The",str(len(cor_feature)), 'most important features are:', cor_feature)


#%% data input, output, train and test for classification
#define failure
df_net =pd.DataFrame( np.where(df_t_scaled < -0,True, False), index = df_t_scaled.index,columns = ['net_loss'] ).astype(int)
df_severe =pd.DataFrame( np.where(df_t_scaled < -1,True, False), index = df_t_scaled.index,columns = ['severe_loss'] ).astype(int)
loss_intensity = df_severe
X, y = df_cli2_scaled, loss_intensity
#divide data train and test
X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, loss_intensity, test_size=0.3, random_state=0)
#%% grid search - define quantity of features to be used - ANOVA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
# define number of features to evaluate
num_features = [i+1 for i in range(X.shape[1])]
# enumerate each number of features
results = list()
for k in num_features:
	# create pipeline
	model = LogisticRegression(solver='liblinear')
	fs = SelectKBest(score_func=f_classif, k=k)
	pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
	# evaluate the model
	scores = evaluate_model(pipeline, X, y)
	results.append(scores)
	# summarize the results
	#print('>%d %.3f (%.3f)' % (k, np.mean(scores), np.std(scores)))
dt_feat= pd.DataFrame(results, index =num_features ).T
bplot = sns.boxplot(data=dt_feat, width=0.5,showmeans=True).set(title = "ANOVA f-test classification accuracy for features", ylabel = 'Accuracy', xlabel = 'Number of features')
#%% ANOVA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train.values.ravel(), X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %s: %f' % (X_train.columns[i], fs.scores_[i]))
# plot the scores
# plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
anova = plt.bar(X_train.columns, fs.scores_)
print(X_train.iloc[:,np.argsort(fs.scores_)[-5:]].columns.tolist()[::-1])
#%% mutual information feature selection
# feature selection
def select_features_mutual(X_train, y_train, X_test):
	# configure to select all features
	fs_mutual = SelectKBest(score_func=mutual_info_classif, k='all')
	# learn relationship from training data
	fs_mutual.fit(X_train, y_train.values.ravel())
	# transform train input data
	X_train_fs = fs_mutual.transform(X_train)
	# transform test input data
	X_test_fs = fs_mutual.transform(X_test)
	return X_train_fs, X_test_fs, fs_mutual
 
# feature selection
X_train_fs, X_test_fs, fs_mutual = select_features_mutual(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
mutual_inf = plt.bar(X_train.columns, fs_mutual.scores_)
print(X_train.iloc[:,np.argsort(fs.scores_)[-5:]].columns.tolist()[::-1])

#%% random forest feature selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
# fit the model
rfc.fit(X, y.values.ravel())
# get importance
importance = rfc.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
rf_feat_sel = plt.bar(X_train.columns, importance)
print(X_train.iloc[:,np.argsort(importance)[-5:]].columns.tolist()[::-1])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=144)
fig.tight_layout()
fig.subplots_adjust(hspace=.1)
ax1.bar(X_train.columns, fs.scores_)
ax1.set_xticklabels([])
ax1.set_title("ANOVA")
ax2.bar(X_train.columns, fs_mutual.scores_)
ax2.set_title("Mutual information")
ax2.set_xticklabels([])
ax3.bar(X_train.columns, importance)
ax3.set_title("Random forest classification")

fig.savefig('features_rank_bar.png', bbox_inches='tight')

#%% cross validate the scores on a number of different random splits of the data to assess robustness of data importance
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from collections import defaultdict
def mean_decreasy_accuracy(model,X,y):
    scores = defaultdict(list)
    names =  X.columns.values
    rs = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    for train_idx, test_idx in rs.split(X):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        Y_train, Y_test = y.values.ravel()[train_idx], y.values.ravel()[test_idx]
        r = model.fit(X_train, Y_train)
        acc = f1_score(Y_test, model.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = f1_score(Y_test, model.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print ("Features sorted by their score:")
    print (sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True))

# Use two models, logistic regression and random forest classifier
model_list = [rfc,LogisticRegression(random_state=0, max_iter=10e4)]
for model in model_list:
    mean_decreasy_accuracy(model,X,y)


#%% ROC curve
from sklearn.metrics import confusion_matrix , plot_confusion_matrix, roc_auc_score,auc, roc_curve, precision_recall_curve, fbeta_score
def metrics_fun(X_test, y_test, y_pred, clf, n_features = 'all'):    
    # CONFUSION MATRIX TO ASSESS METRIC
    confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf, X_test, y_test,display_labels=['Negative', 'Positive'],cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix of the classifier - {n_features} feaures')
    plt.show()
    #### ROC Curves
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print(f'\n {n_features} features - ROC AUC=%.3f (No Skill: ROC AUC=%.3f)' % (lr_auc,ns_auc))
    # calculate ROC curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC - {n_features} feaures')   
    plt.legend()
    plt.show()
    
    ### precision-recall / f1 score, auc score - better for highly skewed cases
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
    lr_f2= fbeta_score(y_test, y_pred, beta=2)
    # summarize scores
    print('SCORES: f1=%.3f ; f2=%.3f; PR-AUC=%.3f' % (lr_f1, lr_f2, lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test.values==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    plt.title(f'Precision-recall - {n_features} feaures')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
#%% Probabilistic failure  - weighted logistic regression
print("_____________________________________ \n Weighted Logistic Regression")
###### all features
clf = LogisticRegression(random_state=0, max_iter=10e4,class_weight='balanced').fit(X_train, y_train.values.ravel())
importance = np.abs(clf.coef_)[0]
print('\n Coeficients/weights for all features: \n', importance)
print(f"\n All features results for {list(loss_intensity.columns.values)[0]}:")
print(f"{list(loss_intensity.columns.values)[0]} - training score is" , clf.score(X_train, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - test score is" , clf.score(X_test, y_test.values.ravel()))
y_pred = clf.predict(X_test)

######## selecting most important features
number_of_features = 4
sel_states = SelectFromModel(LogisticRegression( random_state=0 ,max_iter=10e4),threshold=-np.inf, max_features = number_of_features)
sel_states.fit(X_train, y_train.values.ravel())
selected_feat_states = X_train.columns[(sel_states.get_support())]
print('\n Number of selected features: {}'.format(len(selected_feat_states)), 'which are', selected_feat_states.values)
#converting data to include selected features
X_train_selected = sel_states.transform(X_train)
X_test_selected = sel_states.transform(X_test)

clf_states = LogisticRegression( random_state=0,max_iter=10e4,class_weight='balanced').fit(X_train_selected, y_train.values.ravel())
y_pred_selected = clf_states.predict(X_test_selected)
score_mean_all = clf_states.score(X_test_selected, y_test.values.ravel())
print(f"Selected features (coef = {clf_states.coef_}) results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_states.score(X_train_selected, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" ,score_mean_all )
scores_cv_mean_all = cross_val_score(clf_states, np.concatenate((X_train_selected, X_test_selected), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=4).mean()
print('5 cross validation score:',scores_cv_mean_all)
## Logistic
#all    
metrics_fun(X_test, y_test, y_pred, clf)
#selected    
metrics_fun(X_test_selected, y_test, y_pred_selected, clf_states, n_features=number_of_features)
print("_____________________________________")

#%% random forest classifier
print("_____________________________________ \n Random Forest")
# all features
clf_rf_all = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 3,class_weight='balanced').fit(X_train, y_train.values.ravel())
score_mean_all_rf = clf_rf_all.score(X_test, y_test.values.ravel())
print(f"All features results: \n", f"{list(loss_intensity.columns.values)[0]} - All training score is" , clf_rf_all.score(X_train, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - All test score is" , score_mean_all_rf )
y_pred = clf_rf_all.predict(X_test)

#select most important ones
sel = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=0), max_features = 5)
sel.fit(X_train, y_train.values.ravel())
selected_feat= X_train.columns[(sel.get_support())]
print("\n Random Forest \n The selected features are",len(selected_feat), selected_feat.values)
# transform
X_train_selected = sel.transform(X_train)
X_test_selected = sel.transform(X_test)
# select features
clf_rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 3, class_weight='balanced').fit(X_train_selected, y_train.values.ravel())
score_mean_sel_rf = clf_rf.score(X_test_selected, y_test.values.ravel())
print(f"Selected features results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_rf.score(X_train_selected, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" , score_mean_sel_rf )
y_pred_selected = clf_rf.predict(X_test_selected)

##### plot the tree
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf_rf.estimators_[0]
feature_list = list(selected_feat.values)
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

from PIL import Image                                                                                
from IPython.display import Image 
pil_img = Image(filename='tree.png')
display(pil_img)

# Get numerical feature importances
importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
  
#all    
metrics_fun(X_test, y_test, y_pred, clf_rf_all)
#selected    
metrics_fun(X_test_selected, y_test, y_pred_selected, clf_rf,n_features=number_of_features)

print("_____________________________________")
#%% BALANCED random forest classifier - Random undersampling of the majority class in reach bootstrap sample. 
from imblearn.ensemble import BalancedRandomForestClassifier
print("_____________________________________ \n Balanced Random Forest")
# all features
clf_brf_all = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 4, min_samples_split=0.05).fit(X_train, y_train.values.ravel())
print(f"All features results: \n", f"{list(loss_intensity.columns.values)[0]} - All training score is" , clf_brf_all.score(X_train, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - All test score is" , clf_brf_all.score(X_test, y_test.values.ravel()) )
y_pred = clf_brf_all.predict(X_test)

#select most important ones
sel = SelectFromModel(BalancedRandomForestClassifier(n_estimators=1000, random_state=0), max_features = 5)
sel.fit(X_train, y_train.values.ravel())
selected_feat= X_train.columns[(sel.get_support())]
print("\n Balanced Random Forest \n The selected features are",len(selected_feat), selected_feat.values)
# transform
X_train_selected = sel.transform(X_train)
X_test_selected = sel.transform(X_test)
# select features
clf_brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 4, min_samples_split=0.05).fit(X_train_selected, y_train.values.ravel())
print(f"Selected features results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_brf.score(X_train_selected, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" , clf_brf.score(X_test_selected, y_test.values.ravel()))
y_pred_selected = clf_brf.predict(X_test_selected)

#### plot the tree
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf_brf.estimators_[1]
feature_list = list(selected_feat.values)
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree_bal.png')
pil_img = Image(filename='tree_bal.png')
display(pil_img)

importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#all    
metrics_fun(X_test, y_test, y_pred, clf_brf_all)
#selected    
metrics_fun(X_test_selected, y_test, y_pred_selected, clf_brf,n_features=3)
print("_____________________________________")

#%% Partial  dependence functions and plots
from sklearn.inspection import plot_partial_dependence

X_sel = pd.DataFrame(X_train_selected)
X_sel.columns = selected_feat
ppd_feat = ['tmx8', 'spei8', 'spei9']


# plot_sing = plot_partial_dependence(clf_brf, X_sel, ppd_feat) 

# plot_duo = plot_partial_dependence(clf_brf, X_sel, [('tmx8', 'spei8')]) 

# plot_all = plot_partial_dependence(clf_brf_all, X_train, X_train.columns) 

# fig, (ax1) = plt.subplots(1, 1, figsize=(8, 16), dpi=150)
# fig.tight_layout()
# plot_all.plot(ax=ax1)
# ax1.set_title("Partial dependence of soybean yield failure probability on features")
# fig.savefig('all_partial_plots.png', bbox_inches='tight')

# #create the image
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), dpi=150)
# plot_sing.plot(ax=ax1)
# ax1.set_title("Partial dependence of soybean yield failure probability on features")
# plot_duo.plot(ax=ax2)
# ax2.set_title("Partial dependence of soybean yield failure probability on two features")
# fig.savefig('partial_plot.png', bbox_inches='tight')
# # pil_img = Image(filename='partial_plot.png')
# # display(pil_img)


#%% gridsearchCV RFbalanced

# model = BalancedRandomForestClassifier()
# n_estimators = [500, 1000]
# max_depth = [2, 3,4]
# min_samples_split=[0.01, 0.05, 0.1]
# # define grid search
# grid = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1')
# grid_result = grid_search.fit(X, y)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    
#%% See how the probability function is working
# from sklearn.metrics import brier_score_loss
# from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# predictions_rf = clf_brf_all.predict_proba(X_test)
# predictions_rf_unbalanced = clf_rf_all.predict_proba(X_test)

# clf_brf_all_2 = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 4, min_samples_split=0.05)

# clf_isotonic = CalibratedClassifierCV(clf_brf_all_2, method='isotonic')
# clf_isotonic.fit(X_train, y_train.values.ravel())
# predictions_isotonic = clf_isotonic.predict_proba(X_test)

# bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
## Here we are binning the predictions from the random forest prediction 
## and then looking at the percentage of each bin that is assigned to the positive class
# predictions_df = pd.DataFrame(predictions_rf)
# predictions_df['actual'] = y_test.values
# predictions_df['bin'] = pd.cut(predictions_df[1], bins)
# predictions_df['bin_mid'] = predictions_df['bin'].apply(lambda x: x.mid).astype(float)
# bin_df = predictions_df.groupby('bin_mid').actual.mean().reset_index()

# predictions_un_df = pd.DataFrame(predictions_rf_unbalanced)
# predictions_un_df['actual'] = y_test.values
# predictions_un_df['bin'] = pd.cut(predictions_un_df[1], bins)
# predictions_un_df['bin_mid'] = predictions_un_df['bin'].apply(lambda x: x.mid).astype(float)
# bin_un_df = predictions_un_df.groupby('bin_mid').actual.mean().reset_index()

## Here we are binning the predictions from the isotonic calibration
## and then looking at the percentage of each bin that is assigned to the positive class
# predictions_isotonic_df = pd.DataFrame(predictions_isotonic)
# predictions_isotonic_df['actual'] = y_test.values
# predictions_isotonic_df['bin'] = pd.cut(predictions_isotonic_df[1], bins)
# predictions_isotonic_df['bin_mid'] = predictions_isotonic_df['bin'].apply(lambda x: x.mid).astype(float)
# bin_isotonic_df = predictions_isotonic_df.groupby('bin_mid').actual.mean().reset_index()

## Finally, plot the binned predictions
# plt.figure(figsize=(10,10))
# plt.plot(bin_df.bin_mid, bin_df.actual, '.-');
# plt.plot(bin_un_df.bin_mid, bin_un_df.actual, '.-');
# # plt.plot(bin_isotonic_df.bin_mid, bin_isotonic_df.actual, '.-');
# plt.plot([0,1], [0,1], 'k')
# plt.legend(['Balanced Random Forest', 'Unbalanced Random Forest', 'x=y']);
# plt.xlabel('Predicted probability of positive class');
# plt.ylabel('Fraction belonging to positive class');
# plt.title('Calibration Curves - Random Forest');

# print("Brier scores: (smaller is better)")

# clf_score = brier_score_loss(y_test, predictions_rf[:,1])
# print("No calibration: %1.3f" % clf_score)

# clf_isotonic_score = brier_score_loss(y_test, predictions_isotonic[:,1])
# print("With isotonic calibration: %1.3f" % clf_isotonic_score)


# metrics_fun(X_test, y_test, clf_isotonic.predict(X_test), clf_isotonic)

#%% GRADIENT BOOST STILL TESTING
from sklearn.ensemble import GradientBoostingClassifier
# define the model

model = GradientBoostingClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [500,1000]
grid['learning_rate'] = [0.75,0.1]
grid['subsample'] = [0.5, 0.7, 1.0]
grid['max_depth'] = [2, 3]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1')
# execute the grid search
grid_result = grid_search.fit(X, y.values.ravel())
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.75, max_depth=3, subsample = 1, random_state=0)
clf.fit(X_train, y_train.values.ravel())

metrics_fun(X_test, y_test, clf.predict(X_test), clf)