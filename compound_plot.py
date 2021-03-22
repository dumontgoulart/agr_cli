# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:58:51 2021

@author: morenodu
"""


df_features_ec_season, df_features_ec_season_2C = df_features_ec_season_us, df_features_ec_season_2C_us

y_pred = brf_model_us.predict(df_features_ec_season)
y_pred_2C = brf_model_us.predict(df_features_ec_season_2C)
# Plot graphs comparing the difference between 2C and PD
for (df_features_ec_season_1,df_features_ec_2C_season_1) in zip([df_features_ec_season],
                                                                [df_features_ec_season_2C]):
    
    df_features_ec_season_fail_PD =pd.concat([
        df_features_ec_season_1, pd.DataFrame(
            np.array([y_pred < -1]).T,
            index=df_features_ec_season_1.index, columns=['Scenario'])],axis=1)
    
    df_features_ec_season_fail_PD['Scenario'] = 'PD'
    
    df_features_ec_season_fail_PD_t = df_features_ec_season_fail_PD[y_pred == 1]
    df_features_ec_season_fail_PD_t['Scenario'] = 'Failure PD'
    
    df_features_ec_season_fail_2C =pd.concat([
        df_features_ec_2C_season_1, pd.DataFrame(
            np.array([y_pred_2C > -1 ]).T,
            index=df_features_ec_2C_season_1.index,columns=['Scenario'])],axis=1)
    
    df_features_ec_season_fail_2C['Scenario'] = '2C'
    
    df_features_ec_season_fail_2C_t = df_features_ec_season_fail_2C[y_pred_2C == 1]
    df_features_ec_season_fail_2C_t['Scenario'] = 'Failure 2C'
                
    df_features_ec_season_scenarios = pd.concat([
        df_features_ec_season_fail_PD, df_features_ec_season_fail_2C], axis= 0)
    
    df_features_ec_season_scenarios_t = pd.concat([
        df_features_ec_season_fail_PD_t, df_features_ec_season_fail_2C_t], axis= 0)
    
    for (y_axis, x_axis) in zip([1,0,0],[2,2,1]): 
        # sns.lmplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis], 
        #            x=df_features_ec_season.columns[x_axis],fit_reg=True, 
        #            scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
        # plt.title("Climatic variables for each scenario")
        
        plt.figure(figsize=(10,10))
        g = sns.JointGrid()
        g1 = sns.kdeplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis],
                      x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, alpha= 0.7, ax=g.ax_joint)
        
        g2 = sns.kdeplot(data=df_features_ec_season_scenarios_t, y=df_features_ec_season_scenarios_t.columns[y_axis],
                      x=df_features_ec_season_scenarios_t.columns[x_axis],hue='Scenario',fill=True, alpha= 0.7, ax=g.ax_joint)
        
        g3 = sns.kdeplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_y)
        g4 = sns.kdeplot(data=df_features_ec_season_scenarios, x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_x)
        g5 = sns.kdeplot(data=df_features_ec_season_scenarios_t, y=df_features_ec_season.columns[y_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_y)
        g6 = sns.kdeplot(data=df_features_ec_season_scenarios_t, x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_x)
        
        plt.show()
        # plt.title("Climatic variables for each scenario")
        
        plt.figure(figsize=(10,10))
        g = sns.JointGrid()
        g1 = sns.kdeplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis], palette = ["#92C6FF", "#fabbff"],
                      x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, alpha= 0.7, ax=g.ax_joint)
        
        # g2 = sns.kdeplot(data=df_features_ec_season_scenarios_t, y=df_features_ec_season_scenarios_t.columns[y_axis],palette = ["#97F0AA","#FF9F9A"],
                      # x=df_features_ec_season_scenarios_t.columns[x_axis],hue='Scenario',fill=True, alpha= 0.7, ax=g.ax_joint)
        
        g3 = sns.kdeplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis],hue='Scenario',palette = ["#92C6FF", "#fabbff"],fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_y)
        g4 = sns.kdeplot(data=df_features_ec_season_scenarios, x=df_features_ec_season.columns[x_axis],hue='Scenario',palette = ["#92C6FF", "#fabbff"],fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_x)
        g5 = sns.kdeplot(data=df_features_ec_season_scenarios_t, y=df_features_ec_season.columns[y_axis],hue='Scenario',palette = ["#97F0AA","#FF9F9A"],fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_y)
        g6 = sns.kdeplot(data=df_features_ec_season_scenarios_t, x=df_features_ec_season.columns[x_axis],hue='Scenario',palette = ["#97F0AA","#FF9F9A"],fill=True, legend=False, alpha= 0.7, ax=g.ax_marg_x)
        
        plt.show()
        
        
        
        g5 = sns.kdeplot(data=df_features_ec_season_scenarios, x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7)
        g6 = sns.kdeplot(data=df_features_ec_season_scenarios_t, x=df_features_ec_season.columns[x_axis],hue='Scenario',fill=True, legend=False, alpha= 0.7)
        