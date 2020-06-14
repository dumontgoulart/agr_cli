import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from mask_shape_border import mask_shape_border

# Yield model
DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-61.25,-44.25),lat=slice(-5.25,-32.75), time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = mask_shape_border(DS_y,'gadm36_BRA_0.shp' )
df_t=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore

#climate
DS_cli=xr.open_dataset("temp_evp_prec_era5_monthly.nc").sel(time=slice('1979-09-01','2010-03-31'),longitude=slice(-61.25,-44.25),latitude=slice(-5.25,-32.75))
DS_cli= mask_shape_border(DS_cli,'gadm36_BRA_0.shp' ) #mask data for country's shape
DS_cli['t2m']=DS_cli.t2m -273.15
DS_cli['tp']=DS_cli.tp * 1000
DS_cli['e']=DS_cli.e * 1000
DS_cli.t2m.attrs = {'units': 'Celcius degree', 'long_name': '2 metre temperature'}
DS_cli.tp.attrs = {'units': 'mm', 'long_name': 'Total precipitation'}
DS_cli.e.attrs = {'units': 'mm of water equivalent', 'long_name': 'Evaporation', 'standard_name': 'lwe_thickness_of_water_evaporation_amount'}
DS_cli['growing_month'] = DS_cli["time.month"] # crop season standards
DS_cli['growing_year'] = DS_cli["time.year"]
for i in range(len(DS_cli["time"].values)):
    if DS_cli["time.month"].values[i] < 9:
        DS_cli['growing_year'][i] = DS_cli["time.year"][i]
    else:
        DS_cli["growing_year"][i] = DS_cli["time.year"][i]+1    
DS_cli['growing_month'].values = np.array(list(np.arange(1,8))*(DS_cli['time.year'].values[-1]-DS_cli['time.year'].values[0]))

df_temp=DS_cli.t2m.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_prec=DS_cli.tp.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_e=DS_cli.e.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
#transform in matrix for machine learning 
mat_temp = df_temp.values.reshape(31,7) # divide by each month 
mat_prec = df_prec.values.reshape(31,7) # divide by each month
mat_e = df_e.values.reshape(31,7) # divide by each month
mat_cli= np.concatenate((mat_temp, mat_prec, mat_e), axis=1)
df_cli2 = pd.DataFrame(mat_cli, index = df_t.index )
column_names=[]
for i in ['temp', 'prec', 'evap']:
    for j in range(1,8):
        column_names.append(i+str(j))
df_cli2.columns = column_names
df_total = pd.concat([df_cli2,df_t], axis=1, sort=False)

#%% fit linear regressions and plot them compared with the scatter plots and the respective R2 score:
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def scatter_plot(feature, target):
    linear_regressor = LinearRegression()
    linear_regressor.fit(df_cli2[feature].values.reshape(-1,1), df_t[target])
    Y_pred = linear_regressor.predict(df_cli2[feature].values.reshape(-1,1))
    score = format(linear_regressor.score(df_cli2[feature].values.reshape(-1,1), df_t[target]),'.3f')

    plt.figure(figsize=(10,6))
    plt.scatter(df_cli2[feature], df_t[target], c='black')
    plt.title(f"R2 (0-1) score is {score}", fontsize=20)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel("Yield ton/ha", fontsize=16)
    plt.plot(df_cli2[feature], Y_pred, color='red')
    return score

score_set=[]
for i in df_cli2.columns.values:
    score_i = scatter_plot(i,'yield')
    score_set.append(float(score_i))

sc_set=pd.DataFrame(index = column_names,data = score_set, columns=['R2_score'])
print('The maximum score is', sc_set.max().values, ', corresponding to the feature:', sc_set.R2_score.idxmax())
sc_set.sort_values(by=['R2_score'], ascending=False)

#%% Regularize/standard data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler=StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_total), columns = df_total.columns, index=df_total.index)
df_cli2_scaled = pd.DataFrame(scaler.fit_transform(df_cli2), columns = df_cli2.columns, index=df_cli2.index)
df_t_scaled = pd.DataFrame(scaler.fit_transform(df_t),columns = df_t.columns, index=df_t.index)

X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, df_t_scaled, test_size=0.4, random_state=0)

#%% correlation around features and yield
import seaborn as sns

corrmat = df_scaled.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,10))
g = sns.heatmap(df_scaled[top_corr_features].corr(),annot=True, cmap="RdYlGn")

#%% RANDOM FOREST first trial
from sklearn.metrics import mean_squared_error
# All features
regr = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
regr.fit(X_train, y_train.values.ravel())
print(regr.n_features_,regr.feature_importances_)
print(regr.score(X_train, y_train))
print(regr.score(X_test, y_test))
# Select most important features
sel = SelectFromModel(RandomForestRegressor(n_estimators = 10000))
sel.fit(X_train, y_train.values.ravel())
selected_feat= X_train.columns[(sel.get_support())]
print('Number of features selected:', len(selected_feat), selected_feat.values)

X_important_train = sel.transform(X_train)
X_important_test = sel.transform(X_test)
# Apply model for most important features
clf_important = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
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
from sklearn.ensemble import RandomForestRegressor

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
 
#%% lasso feature selection
clf = LassoCV(max_iter=10e8).fit(X_train, y_train.values.ravel())
print(df_cli2.columns[(clf.coef_ !=0)])
print("training score ", clf.score(X_train,y_train))
print("test score ", clf.score(X_test,y_test))

# LASSO manual selection and alpha calibration
lasso = Lasso(max_iter=10e8)
parameters = {'alpha': [0.001, 0.01,0.02,0.05,0.07,0.1] }

lasso_regressor = GridSearchCV(lasso, parameters, cv=10)
lasso_regressor.fit(X_train,y_train.values.ravel())
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

lasso001 = Lasso(alpha=0.1, max_iter=10e8)
lasso001.fit(X_train,y_train)
pred = lasso001.predict(X_test)

train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print ("number of features used: for alpha = 0.1:", coeff_used001, df_cli2.columns[(lasso001.coef_!=0)].values)
print ("training score for alpha = 0.1:", train_score001 )
print ("test score for alpha = 0.1: ", test_score001)


lasso005 = Lasso(alpha=0.05, max_iter=10e8)
lasso005.fit(X_train,y_train)
pred = lasso005.predict(X_test)

train_score005=lasso005.score(X_train,y_train)
test_score005=lasso005.score(X_test,y_test)
coeff_used005 = np.sum(lasso005.coef_!=0)
print ("number of features used: for alpha = 0.01:", coeff_used005, df_cli2.columns[(lasso005.coef_!=0)].values)
print ("training score for alpha = 0.01:", train_score005 )
print ("test score for alpha = 0.01: ", test_score005)


#%%# Sequential Forward Floating Selection(sffs) or backward elemination
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, df_t_scaled, test_size=0.33, random_state=0)


sffs = SFS(LinearRegression(), 
          k_features=(2,11), 
          forward=False, 
          floating=True,
          scoring = 'neg_mean_squared_error',
          cv=0)
sffs.fit(X_train, y_train)
sffs.k_feature_names_
# plot figure showing features improvement
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sffs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

sffs = SFS(LinearRegression(), 
          k_features=(3), 
          forward=False, 
          floating=True,
          scoring = 'r2',
          cv=0)
sffs.fit(X_train, y_train)
print(sffs.k_feature_names_)

X_train_sfs = sffs.transform(X_train)
X_test_sfs = sffs.transform(X_test)
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_sfs, y_train)

y_pred = linear_regressor.predict(X_test_sfs)
print(linear_regressor.score(X_train_sfs, y_train))
print(linear_regressor.score(X_test_sfs, y_test))

#%% not working, not good for regression?
from sklearn.feature_selection import RFE
X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, df_t_scaled, test_size=0.3, random_state=0)
model = LinearRegression()
for i in range(2,8):
    rfe = RFE(model, n_features_to_select=i)
    rfe1 = rfe.fit(X_train, y_train.values.ravel())
    print( df_cli2.columns[(rfe.support_ == True)])
    print(rfe1.score(X_train, y_train.values.ravel()))
    print(rfe1.score(X_test, y_test.values.ravel()))
    
    
#%% Once more LASSO, + BIC< AIC
   
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

EPSILON = 1e-4

# LassoLarsIC: least angle regression with BIC/AIC criterion

model_bic = LassoLarsIC(criterion='bic',max_iter=10e8)
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
plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)







