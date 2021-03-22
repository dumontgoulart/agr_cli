from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
from IPython.display import display

def shap_prop(df_cli2_scaled, df_t_scaled,clf_brf_all): 
    '''
    Function to explore the random forest decision mechanism.
    
    It consists in using the Shapley approach. Here we have the main contributors, the dependence plots and the decision triggers.
    
    Input: climatic features, yield output, model (random forest)
    '''    
    
    df_severe =pd.DataFrame( np.where(df_t_scaled < df_t_scaled.mean()-df_t_scaled.std(),True, False), index = df_t_scaled.index,columns = ['severe_loss'] ).astype(int)
    loss_intensity = df_severe
    X, y = df_cli2_scaled, loss_intensity
    #divide data train and test
    # X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, loss_intensity, test_size=0.3, random_state=0)
    
    #train explainer shap
    explainer = shap.TreeExplainer(clf_brf_all)
    shap_values = explainer.shap_values(X, approximate=False, check_additivity=True)

    # train for bars and scatters
    explainer_dif = shap.TreeExplainer(clf_brf_all, X)
    shap_values_dif = explainer_dif(X)
    
    # get just the explanations for the positive class
    shap_values_dif_one = shap_values_dif[...,1]
    
    # Summary plots
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values[1], X, plot_type="bar")
    shap.summary_plot(shap_values[1], X) # Failure
    
    # bar plot priority
    # shap.plots.bar(shap_values_dif_one) # - not sure why it is giving different results
    
    # plots for dependence plots and scatter + interaction
    # for feature in X_train.columns.values.tolist():
    #     shap.dependence_plot(feature, shap_values[1], X_train, interaction_index=None)
        
    for name in X.columns:
        shap.dependence_plot(name, shap_values[1], X)
    
    # HTML to interact with all predictors
    shap_display_all = shap.force_plot(explainer.expected_value[1], shap_values[1], X, show=False)
    shap.save_html("index.html", shap_display_all) ## open browser for the interactive model
    
    # Decision plots explaining decisions to classify
    shap.decision_plot(explainer.expected_value[1], shap_values[1], X)
    shap.decision_plot(explainer.expected_value[1], shap_values[1][52], X.loc[[2012]]) #2012 year
    shap.decision_plot(explainer.expected_value[1], shap_values[1][53], X.loc[[2013]]) #2012 year
    
    # Calculate force plot for a given value 2012
    shap.initjs() 
    shap_values_2012 = explainer.shap_values( X.loc[[2012]])
    shap_display = shap.force_plot(explainer.expected_value[1], shap_values_2012[1], X.loc[[2012]],matplotlib=True)
    shap_display2013 = shap.force_plot(explainer.expected_value[1],explainer.shap_values( X.loc[[2013]])[1], X.loc[[2013]],matplotlib=True)
    display(shap_display)
    
    #another waterfall for 2012
    # shap.plots.waterfall(shap_values_dif_one[52])
    # shap.plots.waterfall(shap_values_dif_one[53])
