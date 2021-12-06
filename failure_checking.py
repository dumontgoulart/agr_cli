# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:56:24 2021

@author: morenodu
"""
df_features_ec_season = df_features_ec_season_ar

y_pred_2012 = brf_model_ar.predict_proba(df_clim_2012_ar.values.reshape(1, -1))[0][1]
def predictions(brf_model,df_features_ec_season):
        
        y_pred = brf_model.predict(df_features_ec_season)
        score_prc = sum(y_pred)/len(y_pred) 
        print("\n The total failures are:", sum(y_pred),
              " And the ratio of failure seasons by total seasons is:", score_prc, "\n")     
        probs = brf_model.predict_proba(df_features_ec_season)        
        if df_clim_2012_ar is not None:
            seasons_over_2012 = df_features_ec_season[probs[:,1]>= y_pred_2012]
            print(f"\n Number of >= {y_pred_2012} probability failure events: {len(seasons_over_2012)} and mean conditions are:", 
                  np.mean(seasons_over_2012))
        
        return y_pred, score_prc, probs, seasons_over_2012
    
  
JO_fail_2012 = sum( np.where( 
                (df_features_ec_season.iloc[:,0] >=  df_clim_2012_ar.iloc[0]) & 
                (df_features_ec_season.iloc[:,1] >=  df_clim_2012_ar.iloc[1]) &
                (df_features_ec_season.iloc[:,2] <= df_clim_2012_ar.iloc[2]), 1, 0) )
print("\n Number of seasons with conditions equal or more restrict than the 2012 season:", JO_fail_2012)

tmx_2012_analogues = sum(np.where((df_features_ec_season.iloc[:,0] >=  df_clim_2012_ar.iloc[0]), 1, 0))
dtr_2012_analogues = sum(np.where((df_features_ec_season.iloc[:,1] >=  df_clim_2012_ar.iloc[1]), 1, 0))
precip_2012_analogues = sum(np.where((df_features_ec_season.iloc[:,2] <=  df_clim_2012_ar.iloc[2]), 1, 0))
  
    
y_pred, score_prc, probs, seasons_over_2012 = predictions(brf_model_ar, df_features_ec_season_ar)



jo_fails = np.where((df_features_ec_season.iloc[:,0] >=  df_clim_2012_ar.iloc[0]) & 
                (df_features_ec_season.iloc[:,1] >=  df_clim_2012_ar.iloc[1]) &
                (df_features_ec_season.iloc[:,2] <= df_clim_2012_ar.iloc[2]), 1, 0)

# WHich seasons are failure according to JO?
df_features_ec_season[jo_fails == 1 ]

'''
Storyline season
tmx_1_2_3       30.376984
dtr_1_2_3       14.625722
precip_1_2_3    63.320892
'''

# Which seasons are failure according to the RF?
df_features_ec_season[probs[:,1] >= y_pred_2012 ]

# The RF sees a season more extreme and applies a lower confidence level fo failure -> Bad training
probs[24,1]
probs[240,1]
