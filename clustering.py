import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as plticker

import seaborn as sns
from  scipy import signal 
from mask_shape_border import mask_shape_border
from detrend_dataset import detrend_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, f_classif,  SelectKBest

# Crop space to either US or soy states
usa = gpd.read_file('gadm36_USA_1.shp', crs="epsg:4326") 
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio', 'South Dakota','North Dakota', 'Missouri','Arkansas']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Mato Grosso','Rio Grande do Sul','Paraná']
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]

#%% yield model - WOFOST

DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = DS_y.reindex(lat=DS_y.lat[::-1])
DS_y = mask_shape_border(DS_y,soy_us_states ) #clipping for us
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')

df_wofost=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore

#%% climate CRU
DS_t_mean=xr.open_dataset("cru/cru_tmp.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_t_max=xr.open_dataset("cru/cru_tmx.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_t_min=xr.open_dataset("cru/cru_tmn.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_prec=xr.open_dataset("cru/cru_pre.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_evap=xr.open_dataset("cru/cru_vap.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_wet=xr.open_dataset("cru/cru_wet.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_spei = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))

DS_cli = xr.merge([DS_prec.pre,DS_t_max.tmx,DS_evap.vap,DS_wet['wet'].dt.days, DS_spei.spei]).sel(time=slice('1980-01-01','2010-12-31'))
DS_cli_us = mask_shape_border(DS_cli, soy_us_states) #US-shape
DS_cli_us = DS_cli_us.dropna(dim = 'lon', how='all')
DS_cli_us = DS_cli_us.dropna(dim = 'lat', how='all')

#%% clustering
from sklearn.cluster import AgglomerativeClustering
#crop data
da_yield_cropped = DS_y['yield'].dropna(dim = 'lon', how='all')
da_yield_cropped = da_yield_cropped.dropna(dim = 'lat', how='all')
da_yield_cropped = da_yield_cropped.fillna(1000)
#cluster it for 10 categoris + nan (1000)
hor_data = da_yield_cropped.stack(z=("lat", "lon"))
clustering = AgglomerativeClustering(linkage='average', n_clusters = 11).fit(hor_data.T)
hor_dataset = hor_data.to_dataset()
# update the values for lat/lon
hor_dataset['cluster_label'] = (('z'), clustering.labels_)
cluster_yield = hor_dataset['cluster_label'].unstack()
cluster_yield = cluster_yield.where( cluster_yield != hor_dataset['cluster_label'][-1].values)
unique_values = np.unique(cluster_yield)
unique_values = unique_values[~np.isnan(unique_values)]
cluster_names = ['Cluster '+str(i) for i in range(len(unique_values))]

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
cluster_yield.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='tab20')
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

# detrend feature
def detrend_feature_iso(dataarray_reference, dataarray_in, reference_value, NA_value, months_selected):
    dataarray_iso = dataarray_in.where(dataarray_reference > reference_value, NA_value)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=dataarray_in.attrs, name = dataarray_in.name ) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin(months_selected)) 
    return dataarray_iso_2

#convert to dataframe, reshape so every month is in a separate colum:
def reshape_data(dataarray):  #converts and reshape data
    dataframe = dataarray.to_dataframe().dropna(how='all')
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    dataframe = dataframe.reorder_levels(['time', 'year','month', 'lat', 'lon'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe

# Features considered for this case
list_feature_names = list(DS_cli_us.keys())
column_names=[]
for i in list_feature_names:
    for j in range(6,11):
        column_names.append(i+str(j))

# calculating for each cluster the training matrix features + target:
all_clusters_partitioned = np.zeros((len(unique_values)), dtype=object)
for cluster in range(len(unique_values)): #for each cluster, define yield and climate features b*X=y
    # yield - target(y)
    DS_y_c = DS_y['yield'].where(cluster_yield == unique_values[cluster])
    df_y_c = DS_y_c.to_dataframe().dropna(how='all')
    
    # climate features(X)
    df_features_list = []
    for feature in list_feature_names:        
        DS_cluster_feature = DS_cli_us[feature].where(cluster_yield == unique_values[cluster])
        da_det = detrend_feature_iso((DS_cli_us.tmx.where(cluster_yield == unique_values[cluster])), DS_cluster_feature,-300, -30000, [6,7,8,9,10])
        df_cluster_feature = reshape_data(da_det)
        df_features_list.append(df_cluster_feature)
    
    df_clim_features = pd.concat(df_features_list, axis=1)
    df_clim_features.columns = column_names
    # aggregate it all
    all_clusters_partitioned[cluster] = pd.concat([df_clim_features,df_y_c], axis=1, sort=False) #final table with all features + yield
    
##### test one cluster (this case it is number 3 - Illinois) for experiments
df_cli2 = all_clusters_partitioned[3].loc[:,'pre6':'spei10']
#remove wet and precipitation because of multicolinearity with SPEI
df_cli2 = pd.concat([df_cli2.iloc[:,5:15],df_cli2.iloc[:,20:25]], axis=1, sort=False)
df_y_f = pd.DataFrame(all_clusters_partitioned[3].loc[:,'yield'])
df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)

#%% fit linear regressions and plot them compared with the scatter plots and the respective R2 score:

def scatter_plot(dataset, target_y, feature, target):
    linear_regressor = LinearRegression()
    linear_regressor.fit(dataset[feature].values.reshape(-1,1), target_y[target])
    Y_pred = linear_regressor.predict(dataset[feature].values.reshape(-1,1))
    score = format(linear_regressor.score(dataset[feature].values.reshape(-1,1), target_y[target]),'.3f')
    plt.figure(figsize=(10,6))
    plt.scatter(dataset[feature], target_y[target], c='black')
    plt.title(f"R2 (0-1) score is {score}", fontsize=20)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel("Yield ton/ha", fontsize=16)
    plt.axhline(np.mean(target_y[target]))
    plt.axhline(np.mean(target_y[target]) - np.std(target_y[target]),linestyle='--' )
    plt.plot(dataset[feature], Y_pred, color='red')
    return score

score_set=[]
for i in df_cli2.columns.values:
    score_i = scatter_plot(df_cli2,df_y_f, i,'yield')
    score_set.append(float(score_i))

sc_set=pd.DataFrame(index = column_names,data = score_set, columns=['R2_score'])
print('The maximum score is', sc_set.max().values, ', corresponding to the feature:', sc_set.R2_score.idxmax())
print(sc_set.sort_values(by=['R2_score'], ascending=False))

#%% Regularize/standard data

#standardized
scaler=StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_total), columns = df_total.columns, index=df_total.index)
df_cli2_scaled = pd.DataFrame(scaler.fit_transform(df_cli2), columns = df_cli2.columns, index=df_cli2.index)
df_t_scaled = pd.DataFrame(scaler.fit_transform(df_y_f),columns = df_y_f.columns, index=df_y_f.index)

#minmax
df_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df_total), columns = df_total.columns, index=df_total.index)
df_cli2_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df_cli2), columns = df_cli2.columns, index=df_cli2.index)
df_t_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df_y_f),columns = df_y_f.columns, index=df_y_f.index)

df_failures = df_scaled.loc[df_scaled['yield'] <= -1]
df_cond = df_failures[(df_failures <= -0.9) | (df_failures >= 0.9)]
df_cond_nonscaled = df_cli2[df_scaled['yield'] <= -1]

df_30 = df_scaled[df_cli2['tmx8'] > 30 ]
df_30_sc = df_30[(df_30 <= -0.9) | (df_30 >= 0.9)]


df_cat =pd.DataFrame( np.where(df_t_scaled < -1,'Failure',np.where(df_t_scaled > 1,'High', 'Normal')), index = df_t_scaled.index,columns = ['yield_category'] )
df_total_cat = pd.concat([df_scaled,df_cat], axis=1, sort=False)


#%% Heatmap, selection based on Pearson correlation.

# heatmap with the correlation of each feature + yield
corrmat = df_scaled.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (15,12))
g = sns.heatmap(df_scaled[top_corr_features].corr(),annot=True, cmap="RdYlGn")

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations \n", get_top_abs_correlations(df_cli2_scaled, 6))

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
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(df_cli2_scaled, df_t_scaled, 4)
print("The",str(len(cor_feature)), 'most important features are:', cor_feature)

#%% Pairplot for every feature and distribution according to category

palette ={"Extreme Failure":"r","Failure":"C1","High":"g", "Normal":"k"}
fig_pairplot = sns.pairplot(df_total_cat, hue = 'yield_category', palette = palette  )

#%% RANDOM FOREST first trial
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, df_t_scaled, test_size=0.25, random_state=0)
#%%
# All features
regr = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
regr.fit(X_train, y_train.values.ravel())
print(regr.n_features_,regr.feature_importances_)
print(regr.score(X_train, y_train))
print(regr.score(X_test, y_test))
# Select most important features
sel = SelectFromModel(RandomForestRegressor(n_estimators = 1000))
sel.fit(X_train, y_train.values.ravel())
selected_feat= X_train.columns[(sel.get_support())]
print('Number of features selected:', len(selected_feat), selected_feat.values)

X_important_train = sel.transform(X_train)
X_important_test = sel.transform(X_test)
# Apply model for most important features
clf_important = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train.values.ravel())
print("RF training score: " , clf_important.score(X_important_train, y_train))
print("RF test score: " , clf_important.score(X_important_test, y_test))
# predict the training set
yhat = clf_important.predict(X_important_train)
yhat_t = clf_important.predict(X_important_test)
# calculate the error
mse = mean_squared_error(y_train, yhat)
mse_t = mean_squared_error(y_test, yhat_t)
print('MSE: %.3f' % mse)
print('MSE: %.3f' % mse_t)

#%% BIC first trial
from sklearn.model_selection import GridSearchCV
from math import log
# calculate bic for regression
def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic
 
# generate dataset
# define and fit the model on all data
model = LinearRegression()
model.fit(X_train, y_train)
# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X_train)
# calculate the error
mse = mean_squared_error(y_train, yhat)
print('MSE: %.3f' % mse)
# calculate the bic
bic = calculate_bic(len(y_train), mse, num_params)
print('BIC: %.3f' % bic)  
print("Linear test score: " , model.score(X_test, y_test))
  
#%% estimating alpha to LASSO according to AIC and BIC
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

EPSILON = 1e-4
# LassoLarsIC: least angle regression with BIC/AIC criterion
model_bic = LassoLarsIC(criterion='bic',max_iter=10e3)
model_bic.fit(X_train, y_train.values.ravel())
alpha_bic_ = model_bic.alpha_
model_bic.score(X_train, y_train.values.ravel())
model_bic.score(X_test, y_test.values.ravel())

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_train, y_train.values.ravel())
alpha_aic_ = model_aic.alpha_
model_aic.score(X_train, y_train.values.ravel())
model_aic.score(X_test, y_test.values.ravel())

def plot_ic_criterion(model, name, color):
    criterion_ = model.criterion_
    plt.semilogx(model.alphas_ + EPSILON, criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
    plt.axvline(model.alpha_ + EPSILON, color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection ' )
print(alpha_aic_,alpha_bic_) 

#%% lasso feature selection
clf = LassoCV(max_iter=10e8, cv=20).fit(X_train, y_train.values.ravel())
print("Features selected are:",df_cli2.columns[(clf.coef_ !=0)].values)
print("alpha is:",clf.alpha_  )
print("training score is", clf.score(X_train,y_train))
print("test score ", clf.score(X_test,y_test),'\n \n')
   
# test for LASSO without precipitation values
X_drop = X_train.drop(columns=['days6','days7','days8','days9'] ) # or remove wet ['wet6','wet7','wet8','wet9'] or ['prec6','prec7','prec8','prec9']
X_drop_test = X_test.drop(columns=['days6','days7','days8','days9'] )
#test alpha different values
for alpha in [alpha_aic_, alpha_bic_, 0.1,.2,.3,.4]:
    lasso001 = Lasso(alpha=alpha, max_iter=10e3)
    lasso001.fit(X_drop,y_train)
    pred = lasso001.predict(X_drop_test)
    train_score001=lasso001.score(X_drop,y_train)
    test_score001=lasso001.score(X_drop_test,y_test)
    mse = mean_squared_error(y_test, pred)
    # calculate the bic
    bic = calculate_bic(len(y_train), mse, (len(lasso001.coef_) + 1))
    print ( f" The number of features selected for alpha = {lasso001.alpha} is: \"{np.sum(lasso001.coef_!=0)}\". They are:", X_drop.columns[(lasso001.coef_!=0)].values)
    print (f"training score for alpha = {lasso001.alpha}:", train_score001 )
    print (f"test score for alpha = {lasso001.alpha}: ", test_score001)
    print ('MSE on the test data: %.3f' % mse, '; \n BIC on the test data: %.3f' % bic , '\n \n')
    

#%% select features based on lasso and specific number of features and create new train and test sets for new linear model
# for i in range(len(soy_us_states)):
#     soy_us_states.iloc[i]
for number_of_features in [2,3]   : 
    # number_of_features =5
    clf = LassoCV(max_iter=10e3, cv=20).fit(X_drop, y_train.values.ravel())
    importance = np.abs(clf.coef_)
    print('\n \n Coeficients/weights for all features: \n', importance)
    idx_third = importance.argsort()[-(number_of_features+1)]
    threshold = importance[idx_third] + 0.01
    idx_features = (-importance).argsort()[:number_of_features]
    name_features = np.array(X_drop.columns)[idx_features]
    print(f" \n Selected {number_of_features} most important features:",(name_features))
    #apply threshold to the selector in order to get the fixed number of features previously selected
    sfm = SelectFromModel(clf, max_features = 2)
    sfm.fit(X_drop, y_train.values.ravel())
    X_important_train = sfm.transform(X_drop)
    X_important_test = sfm.transform(X_drop_test)
    
    #TEST LINEAR REGRESSION ON SELECTED FEATURES 
    regr = LinearRegression()
    regr.fit(X_important_train, y_train.values.ravel())
    yield_y_pred = regr.predict(X_important_test)
    
    # The coefficients
    print('\n Coefficients/weights for features:', regr.coef_)
    print(f" Training score is" , regr.score(X_important_train, y_train.values.ravel()))
    print(f" Test score is",  regr.score(X_important_test, y_test.values.ravel()))
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, yield_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination r2: %.2f' % r2_score(y_test, yield_y_pred))


#plot 2d scatterplot correlating two variables
plt.figure(figsize=(10,10)) 
plt.title("Two most important features in a linear regression classified according to yield")
feature1 = X_important_train[:, 0]
feature2 = X_important_train[:, 1]
# feature3 = X_important_train[:, 2]
df_cat_train =pd.DataFrame( np.where(y_train < -1,'Failure',np.where(y_train > 1,'High', 'Normal')), index = y_train.index,columns = ['yield_category'] )
sns.scatterplot(x=feature1, y=feature2,s=200, hue = df_cat_train['yield_category'],style=df_cat_train['yield_category'], palette=palette )
plt.axvline(0, linestyle='--', c='k')
plt.axhline(0, linestyle='--',c='k' )
plt.xlabel("Maximum temperature - {}".format(name_features[0]))
plt.ylabel("Two month {}".format(name_features[1]))
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()

#%% CLASSIFICATION PART

# ANOVA test & Select K Best

selector = SelectKBest(f_classif, k=4)
selector.fit(df_cli2_scaled, df_cat.values.ravel())
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
X_indices = df_cli2_scaled.columns.values
plt.figure(figsize = (20,10))
plt.bar(X_indices , scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
plt.title("K best features according to ANOVA test")
print ('The best parameters for ANOVA test are ', df_cli2_scaled.columns[(selector.get_support())].values)

#%% Select best features using chi2 as metric

from sklearn.feature_selection import chi2

chi_selector = SelectKBest(chi2, k='all')
chi_selector.fit(df_cli2_minmax, df_cat.values.ravel())
chi_support = chi_selector.get_support()
X_new_chi2 = SelectKBest(chi2, k=4).fit(df_cli2_minmax, df_cat.values.ravel())
print ('The best parameters for chi-2 test are ',  df_cli2_minmax.columns[(X_new_chi2.get_support())].values, '\n their respective scores are: ', X_new_chi2.scores_[(X_new_chi2.get_support())], '\n and their p-values are: ', X_new_chi2.pvalues_[(X_new_chi2.get_support())] )

#defining limit value for temperature over 30    
t_30 = pd.concat([pd.DataFrame([(df_cli2.tmx8 > 30).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)
chi_selector.fit(t_30, df_cat.values.ravel())
chi_support = chi_selector.get_support()
print(chi_selector.scores_, chi_selector.pvalues_)

# contingency table
from scipy.stats import chi2_contingency
from scipy.stats import chi2
table = pd.crosstab(t_30.tmx8, t_30.yield_category).values
stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0 - null hypothesis)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0 - null hypothesis)')
else:
	print('Independent (fail to reject H0)')


# loop for the chi2 of each feature - general function
p_list=[]
for feature in df_cli2.columns.values:
    if ord(feature[0]) == ord('t'):
        feat_cat = pd.concat([pd.DataFrame([(df_cli2[feature] > 30).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)
    
    elif ord(feature[0]) == ord('p'):
        feat_cat = pd.concat([pd.DataFrame([df_cli2[feature] < np.percentile(df_cli2[feature], 5).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)

    elif ord(feature[0]) == ord('v'):
        feat_cat = pd.concat([pd.DataFrame([df_cli2[feature] > np.percentile(df_cli2[feature], 95).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)

    elif ord(feature[0]) == ord('d'):
        feat_cat = pd.concat([pd.DataFrame([df_cli2[feature] < np.percentile(df_cli2[feature], 5).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)
    
    elif ord(feature[0]) == ord('s'):
        feat_cat = pd.concat([pd.DataFrame([df_cli2[feature] < np.percentile(df_cli2[feature], 5).astype(int) ]).T,(df_cat == 'Failure').astype(int) ], axis=1, sort=False)
    else:
        print("Error: Feature names changed and are not included.")
    
    chi_selector.fit(feat_cat, df_cat.values.ravel())
    chi_support = chi_selector.get_support()
    print(f'\n Score for feature {feature} is:',chi_selector.scores_[0], chi_selector.pvalues_[0])

    table = pd.crosstab(feat_cat.values[:,0], feat_cat.values[:,1]).values
    stat, p, dof, expected = chi2_contingency(table)
    print('dof=%d' % dof)
    print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
    	print('Positive: Dependent reject H0 (null hypothesis)')
    else:
    	print('Negative: Independent (fail to reject H0)')
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Positive: Dependent: reject H0 (null hypothesis)')
    else:
    	print('Negative: Independent: fail to reject H0 \n \n \n ________________________________')
    p_list.append(p<alpha)

#%% Logistic regression
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold

df_temp_lim = pd.DataFrame( np.where( (df_cli2_scaled.tmx8 > 2) ,'Extreme Heat', np.where(df_cli2_scaled.tmx8 > 1,'Moderate heat',np.where(df_cli2_scaled.tmx8 < -2,'Extreme cold', np.where(df_cli2_scaled.tmx8 < -1,'Cold', 'Normal')))), index = df_cli2_scaled.tmx8.index,columns = ['category'] )
df_net =pd.DataFrame( np.where(df_t_scaled < -0,True, False), index = df_t_scaled.index,columns = ['net_loss'] ).astype(int)
df_severe =pd.DataFrame( np.where(df_t_scaled < -1,True, False), index = df_t_scaled.index,columns = ['severe_loss'] ).astype(int)

# for loss_intensity in [df_net, df_severe]:
loss_intensity = df_severe
X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, loss_intensity, test_size=0.25, random_state=0)
number_of_features = 5
#all features
clf = LogisticRegression( random_state=0,max_iter=10e4).fit(X_train, y_train.values.ravel())
importance = np.abs(clf.coef_)[0]
print('\n Logistic regression \n Coeficients/weights for all features: \n', importance)
print(f"\n All features results for {list(loss_intensity.columns.values)[0]}:")
print(f"{list(loss_intensity.columns.values)[0]} - training score is" , clf.score(X_train, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - test score is" , clf.score(X_test, y_test.values.ravel()))


# selecting most important features
sel_states = SelectFromModel(LogisticRegression( random_state=0 ,max_iter=10e4),threshold=-np.inf, max_features = 5)
sel_states.fit(X_train, y_train.values.ravel())
selected_feat_states = X_train.columns[(sel_states.get_support())]
print('\n Number of selected features: {}'.format(len(selected_feat_states)), 'which are', selected_feat_states.values)

X_train_selected = sel_states.transform(X_train)
X_test_selected = sel_states.transform(X_test)
clf_states = LogisticRegression( random_state=0,max_iter=10e4).fit(X_train_selected, y_train.values.ravel())
score_mean_all = clf_states.score(X_test_selected, y_test.values.ravel())
print(f"Selected features (coef = {clf_states.coef_}) results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_states.score(X_train_selected, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" ,score_mean_all )

scores_cv_mean_all = cross_val_score(clf_states, np.concatenate((X_train_selected, X_test_selected), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=4).mean()
print('5 cross validation score:',scores_cv_mean_all)
print("_____________________________________")

#%% random forest classifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

sel = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=0),max_features = 5)
sel.fit(X_train, y_train.values.ravel())
selected_feat= X_train.columns[(sel.get_support())]
print("\n Random Forest \n The selected features are",len(selected_feat), selected_feat.values)

X_train_selected = sel.transform(X_train)
X_test_selected = sel.transform(X_test)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 5).fit(X_train_selected, y_train.values.ravel())
score_mean_all_rf = clf_rf.score(X_test_selected, y_test.values.ravel())
print(f"Selected features results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_rf.score(X_train_selected, y_train.values.ravel()))
print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" ,score_mean_all_rf )

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
# Get numerical feature importances
importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

print("_____________________________________")

#%% Adaboost
clf_ada = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train_selected, y_train.values.ravel())
print('\n Adaboost score:',clf_ada.score(X_test_selected, y_test.values.ravel()))
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
n_scores = cross_val_score(clf_ada, X_train_selected, y_train.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# Pull out one tree from the forest
tree = clf_ada.estimators_[0]
feature_list = list(selected_feat.values)
# Export the image to a dot file
export_graphviz(tree, out_file = 'ada.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('ada.dot')
# Write graph to a png file
graph.write_png('ada.png')

importances = list(clf_ada.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
print("_____________________________________")


#%%decision tree
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(random_state=0, max_depth = 3, min_samples_split = 2, min_samples_leaf= 2, min_impurity_decrease = 0.0).fit(X_train_selected, y_train.values.ravel())
# clf_tree = clf_rf.estimators_[0] # taking a perfect traiing tree from RF
print(f"\n Decision tree selected training score is" , clf_tree.score(X_train_selected, y_train.values.ravel()))
print('Decision tree score:',clf_tree.score(X_test_selected, y_test.values.ravel()))
# evaluate the model

# Pull out one tree from the forest
feature_list = list(selected_feat.values)
# Export the image to a dot file
export_graphviz(clf_tree, out_file = 'dtree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('dtree.dot')
# Write graph to a png file
graph.write_png('dtree.png')

scores_cv = cross_val_score(clf_tree, np.concatenate((X_train_selected, X_test_selected), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=5)
print('5 cross validation score:', scores_cv.mean())

print("_____________________________________")


#%% multiple clusters analysis
score_list = []
score_mean_list = []
for i in range(len(all_clusters_partitioned)):
    # import from clusters
    df_cli3 = all_clusters_partitioned[i].loc[:,'pre6':'spei10']
    df_y_f_state = pd.DataFrame(all_clusters_partitioned[i].loc[:,'yield'])
    df_total_state = all_clusters_partitioned[i]
    
    scaler=StandardScaler()
    df_scaled_state = pd.DataFrame(scaler.fit_transform(df_total_state), columns = df_total_state.columns, index=df_total_state.index)
    df_cli3_scaled_state = pd.DataFrame(scaler.fit_transform(df_cli3), columns = df_cli3.columns, index=df_cli3.index)
    df_t_scaled_state = pd.DataFrame(scaler.fit_transform(df_y_f_state),columns = df_y_f_state.columns, index=df_y_f_state.index)
  
    df_severe =pd.DataFrame( np.where(df_t_scaled_state < -1,True, False), index = df_t_scaled_state.index,columns = ['severe_loss'] ).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df_cli3_scaled_state, df_severe, test_size=0.2, random_state=0)
    
    #overall mean case
    print("_______________________________________________")
    print(f"\n * Round  for cluster {i} : * \n")

    #TEST FOR TMX8,SPEI8
    X_train_selected_mean = pd.concat([X_train['tmx8'], X_train['spei8']], axis=1)
    X_test_selected_mean = pd.concat([X_test['tmx8'], X_test['spei8']], axis=1)
    selected_feat_states = ([X_test_selected_mean.columns.values[0], X_test_selected_mean.columns.values[1]])
    
    
    clf_states = LogisticRegression(random_state=0,max_iter=10e5).fit(X_train_selected_mean, y_train.values.ravel())
    score_mean = clf_states.score(X_test_selected_mean, y_test.values.ravel())
    print(f"Selected features for mean general case ({selected_feat_states}) test score is" , score_mean)
    print("Mean coeficients are:", clf_states.coef_)
    scores_cv_means = cross_val_score(clf_states,np.concatenate((X_train_selected_mean, X_test_selected_mean), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=5)
    print('cross validation for mean case score is ',scores_cv_means.mean() )
    score_mean_list.append(scores_cv_means.mean())
    print("_________")
 
    #all features
    clf = LogisticRegression(random_state=0,max_iter=10e5).fit(X_train, y_train.values.ravel())
    print(f"\n All features results for {list(loss_intensity.columns.values)[0]}:")
    print(f"{list(loss_intensity.columns.values)[0]} - training score is" , clf.score(X_train, y_train.values.ravel()))
    print(f"{list(loss_intensity.columns.values)[0]} - test score is" , clf.score(X_test, y_test.values.ravel()))
    scores_cv_all = cross_val_score(clf, np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=5)
    print('cross validation for full case score is ',scores_cv_all.mean() )
    print("_________")

    # selecting most important features
    sel_ = SelectFromModel(LogisticRegression(random_state=0,max_iter=10e5),threshold=-np.inf, max_features = 2)
    sel_.fit(X_train, y_train.values.ravel())
    selected_feat = X_train.columns[(sel_.get_support())]
    print('\n Number of selected features: {}'.format(len(selected_feat)), 'and they are', selected_feat.values)
    
    X_train_selected = sel_.transform(X_train)
    X_test_selected = sel_.transform(X_test)
    clf = LogisticRegression(random_state=0,max_iter=10e5).fit(X_train_selected, y_train.values.ravel())
    final_score_i = clf.score(X_test_selected, y_test.values.ravel())
    
    print("Coeficients are:", clf.coef_)
    print( f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf.score(X_train_selected, y_train.values.ravel()))
    print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" , final_score_i)
    
    scores_cv = cross_val_score(clf, np.concatenate((X_train_selected, X_test_selected), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=5)
    print('cross validation for local case score:',scores_cv.mean())
    score_list.append(scores_cv.mean())
    print("_______________________________________________")
   
sc_all_array = np.repeat(scores_cv_mean_all, len(score_list))
table_score = np.array([ np.array(score_list), np.array(score_mean_list),sc_all_array, (np.array(score_list) - np.array(score_mean_list)), (np.array(score_list) - np.array(sc_all_array))]).T
df_table = pd.DataFrame(table_score, index =cluster_names, columns=(['Optimization - Local','Optimization - Mean','Overall baseline','Difference local - mean', 'Difference local - baseline']) )
print(df_table)

plt.figure(figsize=(12,6)) 
plt.title('R2 Score at a state level')
plt.scatter(x=df_table.index, y='Optimization - Local', data=df_table,sizes=(200, 200), label='Local optimization')
plt.scatter(x=df_table.index, y='Optimization - Mean', data=df_table, label='Local - Features from mean model')
#removed for we are not using mean values here plt.hlines(y=score_mean_all, xmin=df_table.index[0], xmax=df_table.index[-1], label='Mean model reference value')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')




