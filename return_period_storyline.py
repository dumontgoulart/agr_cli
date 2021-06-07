import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def return_period_storyline(df_features_ec_season, df_features_ec_2C_season, df_clim_agg_chosen, 
                            df_jo_2012, df_rf_2012, brf_model, df_clim_2012, df_joint_or_rf, 
                            proof_total, df_features_ec_3C_season = None): #df_features_ec_3C_season = None 
    """
    Return period graph generation accounting for storyline joint occurrence values
    
    Parameters: 
    brf_model: the machine learning model trained for the area;
    df_features_ec_season: climatic features (processed by previous function)
    df_features_ec_season_2C = climatic features for future period
    df_clim_2012 = climatic variables for 2012 season
       
    Returns: 
    Return period plot with storyline years .
        
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    # Predictions for Observed data PD
    thresholds=range(0,101,1)
    y_pred_2012 = brf_model.predict_proba(df_clim_2012.values.reshape(1, -1))[0][1]
    def predictions(brf_model,df_features_ec_season):
        
        y_pred = brf_model.predict(df_features_ec_season)
        score_prc = sum(y_pred)/len(y_pred) 
        print("\n The total failures are:", sum(y_pred),
              " And the ratio of failure seasons by total seasons is:", score_prc, "\n")     
        probs = brf_model.predict_proba(df_features_ec_season)        
        if df_clim_2012 is not None:
            limit_2012 = y_pred_2012
        seasons_over_2012 = df_features_ec_season[probs[:,1]>=limit_2012]
        mean_conditions_similar_2012 = np.mean(seasons_over_2012)
        print(f"\n Number of {limit_2012}% events: {len(seasons_over_2012)} and mean conditions are:", 
             mean_conditions_similar_2012)
        
        return y_pred, score_prc, probs, seasons_over_2012, mean_conditions_similar_2012
    
    def return_period(df, colname):

        # Sort data smallest to largest
        sorted_data = df.sort_values(by=colname)
        # Count total obervations
        n = sorted_data.shape[0]      
        # Add a numbered column 1 -> n to use in return calculation for rank
        sorted_data.insert(0, 'rank', range(1, 1 + n))        
        # Calculate probability
        sorted_data["probability"] = (n - sorted_data["rank"] + 1) / (n + 1)        
        # Calculate return - yearly data, no need to further trnasform
        sorted_data["return-years"] = (1 / sorted_data["probability"])
    
        return(sorted_data)
    
    def plot_probs_failure(probs, probs_perm):
        
        # put them together in the same dataframe for plotting
        probs_agg=pd.DataFrame( [probs[:,1],probs_perm[:,1]]).T
        probs_agg.columns=['Ordered','Permuted']
    
        # plots comparing prediction confidence for obs and perumuted
        probs_agg_melt = probs_agg.melt(value_name='Failure probability').assign(data='Density')
        
        # Compare the number of cases above a failure threshold
        fails_prob_together = np.empty([len(thresholds),2])
        i=0
        for prc in thresholds: 
            fails_prob_together[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),
                                        len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
            i=i+1
        
        # Create dataframe with all failure probabilities for ordered and permuted cases
        df_fails_prob_together = pd.DataFrame( fails_prob_together, index = thresholds, 
                                              columns = probs_agg.columns)
         
        sorted_probs = return_period(probs_agg, 'Ordered')

        return probs_agg, sorted_probs, df_fails_prob_together
    
    # PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
    df_features_ec_season_permuted = df_features_ec_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
        
    # predictions for observed data PD
    y_pred, score_prc, probs, seasons_over_2012, mean_conditions_similar_2012 = predictions(brf_model, df_features_ec_season)
    
    # Predictions for permuted
    y_pred_perm, score_prc_perm, probs_perm, seasons_over_2012_perm, mean_conditions_similar_2012_perm = predictions(brf_model,df_features_ec_season_permuted)

    # 2C PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
    df_features_ec_2C_season_permuted = df_features_ec_2C_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
    
    # predictions for observed data PD
    y_pred_2C, score_prc_2C, probs_2C, seasons_over_2012_2C, mean_conditions_similar_2012_2C = predictions(brf_model, df_features_ec_2C_season)
    
    # Predictions for permuted
    y_pred_2C_perm, score_prc_perm_2C, probs_perm_2C, seasons_over_2012_perm_2C, mean_conditions_similar_2012_perm_2C = predictions(brf_model, df_features_ec_2C_season_permuted)
    
    # plots comparing prediction confidence for obs and perumuted
    probs_agg,sorted_probs, df_fails_prob_together = plot_probs_failure(probs, probs_perm)
       
    # plots comparing prediction confidence for obs and perumuted
    probs_agg_2C, sorted_probs_2C, df_fails_prob_together_2C = plot_probs_failure(probs_2C, probs_perm_2C)
    
    if df_features_ec_3C_season is not None:
        # 3C PERMUTATION ---------- 
        df_features_ec_3C_season_permuted = df_features_ec_3C_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
        
        # predictions for observed data PD
        y_pred_3C, score_prc_3C, probs_3C, seasons_over_2012_3C, mean_conditions_similar_2012_3C = predictions(brf_model, df_features_ec_3C_season)
        
        # Predictions for permuted
        y_pred_3C_perm, score_prc_perm_3C, probs_perm_3C, seasons_over_2012_perm_3C, mean_conditions_similar_2012_perm_3C = predictions(brf_model, df_features_ec_3C_season_permuted)
                  
        # plots comparing prediction confidence for obs and perumuted
        probs_agg_3C, sorted_probs_3C, df_fails_prob_together_3C = plot_probs_failure(probs_3C, probs_perm_3C)
        
        
    ### Order RP for 100 years CRU
    probs_cru = brf_model.predict_proba(df_clim_agg_chosen)[:,1]  
    df_probs_cru=pd.DataFrame( probs_cru, columns = ['Ordered'])
    sorted_probs_cru = return_period(df_probs_cru, 'Ordered')
      
    def return_period_ensemble(probs, ensemble_shape=(100,20)):
        # function that reshapes timeseries to have ensemble 100 years for 20 members
        probs_ec_ensemble = np.reshape(probs[:,1], ensemble_shape)
        probs_ec_ensemble = pd.DataFrame(probs_ec_ensemble)
        
        df_ordered = np.empty([100,20])
        df_return_year = np.empty([100,20])
        
        for i in list(probs_ec_ensemble.columns):
            sorted_probs_ec_ensemble = return_period(probs_ec_ensemble, [i])
            df_ordered[:,i] = sorted_probs_ec_ensemble[i]
            df_return_year[:,i] = sorted_probs_ec_ensemble["return-years"]
        
        df_return_year = pd.DataFrame(df_return_year)
        df_ordered = pd.DataFrame(df_ordered)
        return df_ordered, df_return_year
    
    df_ordered, df_return_year = return_period_ensemble(probs)
        
    # Statistics ensemble // If each obs needs to be ploted: # plt.scatter( x=df_return_year, y=df_ordered)
    ord_min = np.min(df_ordered, axis=1)
    ord_max = np.max(df_ordered, axis=1)
    ord_mean = np.mean(df_ordered, axis=1)
    
    # plt.figure(figsize=(6,6), dpi=150)
    # plt.fill_between(df_return_year[0], ord_min, ord_max,
    #                  facecolor="orange", # The fill color
    #                  color='blue',       # The outline color
    #                  alpha=0.2)          # Transparency of the fill
    # plt.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012 threshold')
    # plt.scatter( x=df_return_year[0], y=ord_mean,label = 'PD data',)
    # sns.scatterplot(data = sorted_probs_cru, x=sorted_probs_cru["return-years"], 
    #                 y=sorted_probs_cru["Ordered"], label = 'Observed data',linewidth=0, color = 'k' )
    # plt.xscale('log')
    # plt.legend(loc="lower right")
    # plt.xlabel('Return period')
    # plt.ylabel('Failure probability')
    # plt.savefig('paper_figures/return_period_ensemble.png', format='png', dpi=250)
    # plt.show()
    
    # Return period for joint occurrence
    JO_PD_rp = 2000/df_jo_2012.iloc[0,0]
    JO_2C_rp = 2000/df_jo_2012.iloc[0,1]
    # JO_3C_rp = 2000/df_jo_2012.iloc[0,2]
    RF_PD_rp = 2000/df_jo_2012.iloc[1,0]
    RF_2C_rp = 2000/df_jo_2012.iloc[1,1]
    # RF_3C_rp = 2000/df_jo_2012.iloc[1,2]
    
    
    ### Plot comparing 2C and PD return periods
    # plt.figure(figsize=(6,6), dpi=150)
    # sns.scatterplot(data = sorted_probs, x=sorted_probs["return-years"], 
    #                 y=sorted_probs["Ordered"], label='PD',linewidth=0 ) 
    # sns.scatterplot(data = sorted_probs_2C, x=sorted_probs_2C["return-years"], 
    #                 y=sorted_probs_2C["Ordered"], label = '2C',linewidth=0 )
    
    # # closest_2012 = np.searchsorted(sorted_probs_2C['Ordered'], y_pred_2012, side='left')
    # # closest_2012 = sorted_probs_2C[sorted_probs_2C['rank']==closest_2012]
    
    # plt.scatter(x=JO_PD_rp, y=y_pred_2012, label = 'JO 2012 PD', linewidth=1 )
    # plt.scatter(x=JO_2C_rp, y=y_pred_2012, label = 'JO 2012 2C', linewidth=1 )

    # if y_pred_2012 is not None:
    #     plt.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012')
    # plt.xscale('log')
    # plt.legend(loc="lower right")
    # plt.xlabel('Return period')
    # plt.ylabel('Failure probability')
    # plt.title("Confidence event and return period for PD and 2C")
    # plt.savefig('paper_figures/return_period_storyline.png', format='png', dpi=500)
    # plt.show()
    
    def mean_max_min(df_ordered):
        ord_min = np.min(df_ordered, axis=1)
        ord_max = np.max(df_ordered, axis=1)
        ord_mean = np.mean(df_ordered, axis=1)
        # ci = 1.96 * np.std(df_ordered, axis = 1) / np.sqrt(20)
        # ord_mean = np.median(df_ordered, axis=1)
        # ord_min = ord_mean - ci
        # ord_max = ord_mean + ci
        return ord_mean, ord_min, ord_max    
    
    # Generate ensemble for other scenarios 
    df_ordered_perm, df_return_year_perm = return_period_ensemble(probs_perm)
    df_ordered_2C, df_return_year_2C = return_period_ensemble(probs_2C)
    df_ordered_perm_2C, df_return_year_perm_2C = return_period_ensemble(probs_perm_2C)
 
    # Reference values for mean, min and max
    ord_mean, ord_min, ord_max = mean_max_min(df_ordered)   
    ord_perm_mean, ord_perm_min, ord_perm_max = mean_max_min(df_ordered_perm)   
    ord_2C_mean, ord_2C_min, ord_2C_max = mean_max_min(df_ordered_2C)
    ord_2C_perm_mean, ord_2C_perm_min, ord_2C_perm_max = mean_max_min(df_ordered_perm_2C)
    
    # 3C
    if df_features_ec_3C_season is not None:
        df_ordered_3C, df_return_year_3C = return_period_ensemble(probs_3C)
        df_ordered_perm_3C, df_return_year_perm_3C = return_period_ensemble(probs_perm_3C)
        ord_3C_mean, ord_3C_min, ord_3C_max = mean_max_min(df_ordered_3C)
        ord_3C_perm_mean, ord_3C_perm_min, ord_3C_perm_max = mean_max_min(df_ordered_perm_3C)
        
    # Figure for ensemble obs and 2C impact #####################################
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), dpi=500)
    
    # First subplot - Return period ensembles PD and observed    
    ax1.fill_between(df_return_year[0], ord_min, ord_max,
                      facecolor="blue", # The fill color
                      color='blue',       # The outline color
                      alpha=0.2)   
    
    ax1.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012 threshold')
    ax1.scatter( x=df_return_year[0], y=ord_mean, label = 'PD data',)
    ax1.scatter(x=sorted_probs_cru["return-years"], y=sorted_probs_cru["Ordered"], 
                label = 'Observed data', color = 'k' )
    ax1.set_xscale('log')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Failure probability')
    ax1.set_xlabel('Return period')
    ax1.set_title('a) Observed data and PD scenario')
    
    # Second subplot - Return period ensembles PD and 2C
    ax2.fill_between(df_return_year[0], ord_min, ord_max,
                      facecolor="blue", # The fill color
                      color='blue',       # The outline color
                      alpha=0.2)    
    
    ax2.fill_between(df_return_year_2C[0], ord_2C_min, ord_2C_max,
                      facecolor='orange', # The fill color
                      color='orange',       # The outline color
                      alpha=0.2)        # Transparency of the fill
    ax2.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012 threshold')
    ax2.scatter( x=df_return_year[0], y=ord_mean,label = 'PD data')
    ax2.scatter( x=df_return_year_2C[0], y=ord_2C_mean,label = '2C data', color= "orange")
    
    if df_features_ec_3C_season is not None:
        ax2.fill_between(df_return_year_3C[0], ord_3C_min, ord_3C_max,
                          facecolor='#f0027f', # The fill color
                          color='#f0027f',       # The outline color
                          alpha=0.2)        # Transparency of the fill
        ax2.scatter( x=df_return_year_3C[0], y=ord_3C_mean,label = '3C data', color= "#f0027f")

    ax2.set_xscale('log')
    ax2.legend(loc="lower right")
    ax2.set_ylabel('Failure probability')
    ax2.set_xlabel('Return period')
    ax2.set_title('b) Climate change effects')
    
    plt.tight_layout()
    plt.savefig('paper_figures/return_period_ensemble_obs_2C.png', format='png', dpi=500)
    plt.show()
    
    
    # Figure for compoundness #################################################
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), dpi=500)
    
    
    ax1.fill_between(df_return_year[0], ord_min, ord_max,
                      facecolor="blue", # The fill color
                      color='blue',       # The outline color
                      alpha=0.2)   
    
    ax1.fill_between(df_return_year_perm[0], ord_perm_min, ord_perm_max,
                      facecolor="orange", # The fill color
                      color='orange',       # The outline color
                      alpha=0.2)        # Transparency of the fill
    ax1.scatter( x=df_return_year[0], y=ord_mean, label = 'Original data',)
    ax1.scatter( x=df_return_year_perm[0], y=ord_perm_mean, label = 'Shuffled data')
    ax1.set_xscale('log')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Failure probability')
    ax1.set_xlabel('Return period')
    ax1.set_title('a) Present day')
    
    # Subplot ensembles at PD and 2C comparing with compoundness
    ax2.fill_between(df_return_year_2C[0], ord_2C_min, ord_2C_max,
                      facecolor="blue", # The fill color
                      color='blue',       # The outline color
                      alpha=0.2)   
    
    ax2.fill_between(df_return_year_perm_2C[0], ord_2C_perm_min, ord_2C_perm_max,
                      facecolor="orange", # The fill color
                      color='orange',       # The outline color
                      alpha=0.2)        # Transparency of the fill
    ax2.scatter( x=df_return_year_2C[0], y=ord_2C_mean, label = 'Original data',)
    ax2.scatter( x=df_return_year_perm_2C[0], y=ord_2C_perm_mean, label = 'Shuffle data')
    ax2.set_xscale('log')
    ax2.legend(loc="lower right")
    ax2.set_ylabel('Failure probability')
    ax2.set_xlabel('Return period')
    ax2.set_title('b) 2C')
    
    plt.tight_layout()
    plt.savefig('paper_figures/return_period_ensemble_PD_perm.png', format='png', dpi=500)
    plt.show()
    
    if df_features_ec_3C_season is not None:
        # Figure for compoundness 3C - Compare PD, 2C and 3C for original and permuted data #################################################
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 6), dpi=500)
        
        
        ax1.fill_between(df_return_year[0], ord_min, ord_max,
                          facecolor="blue", # The fill color
                          color='blue',       # The outline color
                          alpha=0.2)   
        
        ax1.fill_between(df_return_year_perm[0], ord_perm_min, ord_perm_max,
                          facecolor="orange", # The fill color
                          color='orange',       # The outline color
                          alpha=0.2)        # Transparency of the fill
        ax1.scatter( x=df_return_year[0], y=ord_mean, label = 'Original data',)
        ax1.scatter( x=df_return_year_perm[0], y=ord_perm_mean, label = 'Shuffled data')
        ax1.set_xscale('log')
        ax1.legend(loc="lower right")
        ax1.set_ylabel('Failure probability')
        ax1.set_xlabel('Return period')
        ax1.set_title('a) Present day')
        
        
        ax2.fill_between(df_return_year_2C[0], ord_2C_min, ord_2C_max,
                          facecolor="blue", # The fill color
                          color='blue',       # The outline color
                          alpha=0.2)   
        
        ax2.fill_between(df_return_year_perm_2C[0], ord_2C_perm_min, ord_2C_perm_max,
                          facecolor="orange", # The fill color
                          color='orange',       # The outline color
                          alpha=0.2)        # Transparency of the fill
        ax2.scatter( x=df_return_year_2C[0], y=ord_2C_mean, label = 'Original data',)
        ax2.scatter( x=df_return_year_perm_2C[0], y=ord_2C_perm_mean, label = 'Shuffled data')
        ax2.set_xscale('log')
        ax2.legend(loc="lower right")
        ax2.set_ylabel('Failure probability')
        ax2.set_xlabel('Return period')
        ax2.set_title('b) 2C')
        
        ax3.fill_between(df_return_year_3C[0], ord_3C_min, ord_3C_max,
                          facecolor="blue", # The fill color
                          color='blue',       # The outline color
                          alpha=0.2)   
        
        ax3.fill_between(df_return_year_perm_3C[0], ord_3C_perm_min, ord_3C_perm_max,
                          facecolor="orange", # The fill color
                          color='orange',       # The outline color
                          alpha=0.2)        # Transparency of the fill
        ax3.scatter( x=df_return_year_3C[0], y=ord_3C_mean, label = 'Original data',)
        ax3.scatter( x=df_return_year_perm_3C[0], y=ord_3C_perm_mean, label = 'Shuffled data')
        ax3.set_xscale('log')
        ax3.legend(loc="lower right")
        ax3.set_ylabel('Failure probability')
        ax3.set_xlabel('Return period')
        ax3.set_title('c) 3C')
        
        plt.tight_layout()
        plt.savefig('paper_figures/return_period_3C_ensemble_PD_perm.png', format='png', dpi=500)
        plt.show()
        
        
    # Figure for compoundness with different methods #################################################
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), dpi=500)
    labels = ['AND','RF','OR']
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    linewidth_graph=2
    
    # First subplot - Comparison compoundness between models
    ax1.vlines(x - width/2, proof_total, df_joint_or_rf.loc[:,'PD']/2000, linewidth = linewidth_graph)
    ax1.vlines(x + width/2, proof_total, df_joint_or_rf.loc[:,'PD perm']/2000, color = 'orange', linewidth = linewidth_graph)
    ax1.scatter(x - width/2, df_joint_or_rf.loc[:,'PD']/2000, label = 'Original data', s=100)
    ax1.scatter(x + width/2, df_joint_or_rf.loc[:,'PD perm']/2000, label = 'Shuffled data', s=100, color = 'orange')
    # rects1 = ax1.bar(x - width/2, df_jo_2012.iloc[0,:]/2000, width, bottom = proof_total, label = 'Original data', color = '#4c72b0')
    # rects2 = ax1.bar(x + width/2, df_rf_2012.iloc[0,:]/2000, width, bottom = proof_total, label = 'Shuffled data', color = '#55a868')
    ax1.axhline(y=proof_total, color = 'k', label = 'Observed failure ratio')
    ax1.set_ylabel('Failure ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('a) Failure ratio compared to observed data')
    ax1.legend()
    
    width = 0.3  # the width of the bars
    linewidth_graph=2
    # Secod subplot - Compoundness decrease under warmer temperatures
    ax2.axhline(y=1, linestyle = '--', color = 'k', label = 'No compound factor' ) #, 
    ax2.vlines(x - width/3, 1, df_joint_or_rf.loc[:,'PD/ PD perm'], linewidth = linewidth_graph)
    ax2.vlines(x, 1, df_joint_or_rf.loc[:,'2C/ 2C perm'], linewidth = linewidth_graph , color = '#fdbb84')  
    ax2.vlines(x + width/3, 1, df_joint_or_rf.loc[:,'3C/ 3C perm'], linewidth = linewidth_graph , color = '#e34a33')         
    ax2.scatter(x - width/3, df_joint_or_rf.loc[:,'PD/ PD perm'],  label='PD', s=90)
    ax2.scatter(x , df_joint_or_rf.loc[:,'2C/ 2C perm'],  label='2C', s=90, color = '#fdbb84')
    ax2.scatter(x + width/3, df_joint_or_rf.loc[:,'3C/ 3C perm'],  label='3C', s=90, color = '#e34a33')
    ax2.set_ylabel('Compound factor')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_title('b) Compound factor in GW scenarios')
    plt.tight_layout()
    plt.savefig('paper_figures/compoundness_approaches.png', format='png', dpi=500)
    plt.show()
    
    # Figure for 2012 season ##########################################################
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), dpi=500)
    # labels = ['PD','2C','3C']
    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    
    # rects1 = ax1.bar(x - width/2, df_jo_2012.iloc[0,:], width, label='AND', color = '#4c72b0')
    # rects2 = ax1.bar(x + width/2, df_rf_2012.iloc[0,:], width, label='RF', color = '#55a868')
    
    # ax1.set_ylabel('2012 analogues')
    # ax1.set_title('a) 2012 analogues for different approaches')
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(labels)
    # ax1.legend()
    
    # # Second subplot
    # print(mean_conditions_similar_2012_2C)
    # print('this should be a single number',mean_conditions_similar_2012_2C[0])
    # print('this should be a single number',mean_conditions_similar_2012_2C[1])
    # print('this should be a single number',mean_conditions_similar_2012_2C[2])
    
    # labels_b = ['Tmx','DTR','Precip']
    # b = np.arange(len(labels_b))  # the label locations
    # width = 0.35  # the width of the bars
    
    # tmx_dev_2C = (mean_conditions_similar_2012_2C[0]-df_clim_2012[0])
    # dtr_dev_2C = (mean_conditions_similar_2012_2C[1]-df_clim_2012[1])
    # precip_dev_2C = (mean_conditions_similar_2012_2C[2]-df_clim_2012[2])
    # dev_set_2C = [tmx_dev_2C, dtr_dev_2C, precip_dev_2C]
    
    # tmx_dev_3C = (mean_conditions_similar_2012_3C[0]-df_clim_2012[0])
    # dtr_dev_3C = (mean_conditions_similar_2012_3C[1]-df_clim_2012[1])
    # precip_dev_3C = (mean_conditions_similar_2012_3C[2]-df_clim_2012[2])
    # dev_set_3C = [tmx_dev_3C, dtr_dev_3C, precip_dev_3C]
    # print(dev_set_3C)

    # ax2.bar(b - width/2, dev_set_2C, width, label='2C failure raise',color = '#fdbb84')
    # ax2.bar(b[1] - width/2, dev_set_2C[1], width, label='2C failure lower',color = '#99d8c9')
    
    # ax2.bar(b + width/2, dev_set_3C, width, label='3C failure raise', color = '#e34a33')
    # ax2.bar(b[1] + width/2, dev_set_3C[1], width, label='3C failure lower', color = '#2ca25f')

    # ax2.set_xticks(b)
    # ax2.set_xticklabels(labels_b)
    # ax2.set_ylabel('Relative deviation')
    # ax2.legend()
    # ax2.set_title('b) Climatic variables change')
    # fig.tight_layout()
    # plt.savefig('paper_figures/2012_analysis.png', format='png', dpi=500)
    # plt.show()
    
    
    from matplotlib.gridspec import GridSpec
    
    
    fig=plt.figure(figsize=(12, 5), dpi=500)
    
    gs=GridSpec(1,6) # 2 rows, 3 columns
    
    ax1=fig.add_subplot(gs[0,0:3]) # First row, first column
    ax2=fig.add_subplot(gs[0,3:4]) # First row, second column
    ax3=fig.add_subplot(gs[0,4:5]) # First row, third column
    ax4=fig.add_subplot(gs[0,5:6]) # Second row, span all columns
    
    labels = ['PD','2C','3C']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    rects1 = ax1.bar(x - width/2, df_jo_2012.iloc[0,:]/2000, width, label='Event-analogues', color = '#4c72b0')
    rects2 = ax1.bar(x + width/2, df_rf_2012.iloc[0,:]/2000, width, label='Impact-analogues', color = '#55a868')
    
    ax1.set_ylabel('Normalized 2012 analogues')
    ax1.set_title('a) Ratio of 2012 analogues')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Second subplot
    print('Mean conditions of similar events to storyline',mean_conditions_similar_2012_2C)
    
    labels_b = ['Tmx','DTR','Precip']
    b = np.arange(len(labels_b))  # the label locations
    width = 0.35  # the width of the bars
    
    tmx_dev_2C = (mean_conditions_similar_2012_2C[0]-df_clim_2012[0])
    dtr_dev_2C = (mean_conditions_similar_2012_2C[1]-df_clim_2012[1])
    precip_dev_2C = (mean_conditions_similar_2012_2C[2]-df_clim_2012[2])
    dev_set_2C = [tmx_dev_2C, dtr_dev_2C, precip_dev_2C]
    
    tmx_dev_3C = (mean_conditions_similar_2012_3C[0]-df_clim_2012[0])
    dtr_dev_3C = (mean_conditions_similar_2012_3C[1]-df_clim_2012[1])
    precip_dev_3C = (mean_conditions_similar_2012_3C[2]-df_clim_2012[2])
    dev_set_3C = [tmx_dev_3C, dtr_dev_3C, precip_dev_3C]
    print(dev_set_3C)

    ax2.bar(b[0] - width/2, dev_set_2C[0], width, label='2C',color = '#fdbb84')
    ax3.bar(b[1] - width/2, dev_set_2C[1], width, label='2C',color = '#fdbb84') #for green: 99d8c9
    ax4.bar(b[2] - width/2, dev_set_2C[2], width, label='2C',color = '#fdbb84')
    
    ax2.bar(b[0] + width/2, dev_set_3C[0], width, label='3C', color = '#e34a33')
    ax3.bar(b[1] + width/2, dev_set_3C[1], width, label='3C', color = '#e34a33') #for green: 2ca25f
    ax4.bar(b[2] + width/2, dev_set_3C[2], width, label='3C', color = '#e34a33')

    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_ylabel('Temperature deviation (°C)')
    ax2.set_xlabel('Tmx')
    # ax2.set_title('b) Conditions change 2C and 3C')
    # ax2.legend()
    
    ax3.set_xticks([])
    ax3.set_xticklabels([])
    ax3.set_ylabel('Temperature deviation (°C)')
    ax3.set_xlabel('DTR')
    ax3.set_title('b) c) d) Conditions for 2012 analogues at 2C and 3C')
    # ax3.legend()
    
    ax4.set_xticks([])
    ax4.set_xticklabels([])
    ax4.set_ylabel('Precipitation deviation (mm/month)')
    ax4.set_xlabel('Precip')
    # ax4.set_title('d) 2012 analogues for different approaches')
    ax4.legend(loc=(1.05, 0.5))
    
    fig.tight_layout()
    plt.savefig('paper_figures/2012_analysis_clim.png', format='png', dpi=500)
    
    return mean_conditions_similar_2012_2C, mean_conditions_similar_2012_3C

    
    