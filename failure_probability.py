import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, cross_validate,RepeatedStratifiedKFold

#%% Start function

def feature_importance_selection(df_cli2, df_y_f, show_scatter = True, feat_sel = False):

    #%% fit linear regressions and plot them compared with the scatter plots and the respective R2 score:
    if show_scatter == True:    
        def scatter_plot(df_cli2,feature):
            linear_regressor = LinearRegression()
            linear_regressor.fit(df_cli2[feature].values.reshape(-1,1), df_y_f)
            Y_pred = linear_regressor.predict(df_cli2[feature].values.reshape(-1,1))
            score = format(linear_regressor.score(df_cli2[feature].values.reshape(-1,1), df_y_f),'.3f')
        # plot
            plt.figure(figsize=(10,6), dpi=100)
            plt.scatter(df_cli2[feature], df_y_f, c='black')
            plt.title(f"R2 (0-1) score is {score}", fontsize=20)
            plt.xlabel(feature, fontsize=16)
            plt.ylabel("Yield ton/ha", fontsize=16)
            plt.axhline(np.mean(df_y_f.values))
            plt.axhline(np.mean(df_y_f.values) - np.std(df_y_f.values),linestyle='--' )
            plt.plot(df_cli2[feature], Y_pred, color='red')
            return score
        
        score_set=[]
        for i in df_cli2.columns.values:
            score_i = scatter_plot(df_cli2,i)
            score_set.append(float(score_i))
        
        sc_set=pd.DataFrame(index = list(df_cli2.columns),data = score_set, columns=['R2_score'])
        print('The maximum score is', sc_set.max().values, ', corresponding to the feature:', sc_set.R2_score.idxmax())
        print(sc_set.sort_values(by=['R2_score'], ascending=False)) 
    #%% Regularize/standard data
    #standardized
    scaler=StandardScaler()
    # df_cli2_scaled = pd.DataFrame(scaler.fit_transform(df_cli2), columns = df_cli2.columns, index=df_cli2.index)
    df_cli2_scaled = df_cli2
    # # SPEI is already scaled, so it needs to be returned to its original form (could you confirm?)
    # df_cli2_scaled[["spei6", "spei7", "spei8", "spei9", "spei10"]] = df_cli2[["spei6", "spei7", "spei8", "spei9", "spei10"]]
    # df_t_scaled = pd.DataFrame(scaler.fit_transform(df_y_f),columns = df_y_f.columns, index=df_y_f.index)
    df_t_scaled = df_y_f
    df_scaled = pd.concat([df_cli2_scaled,df_t_scaled], axis=1, sort=False)
    
    #%% data input, output, train and test for classification
    #define failure
    # df_net =pd.DataFrame( np.where(df_t_scaled < -0,True, False), index = df_t_scaled.index,columns = ['net_loss'] ).astype(int)
    df_severe =pd.DataFrame( np.where(df_t_scaled < df_t_scaled.mean()-df_t_scaled.std(),True, False), index = df_t_scaled.index,columns = ['severe_loss'] ).astype(int)
    loss_intensity = df_severe
    X, y = df_cli2_scaled, loss_intensity
    #divide data train and test
    X_train, X_test, y_train, y_test = train_test_split(df_cli2_scaled, loss_intensity, test_size=0.3, random_state=0)
    
    #%% heatmap with the correlation of each feature + yield
    corrmat = df_scaled.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize = (7,7),dpi=144)
    g = sns.heatmap(df_scaled[top_corr_features].corr(),annot=True, cmap="RdYlGn",vmin=-1, vmax=1)
    plt.title("Pearson's correlation")
    plt.show()

    # # kendall Rank Correlation
    # corrkendall = df_scaled.corr(method='kendall')
    # plt.figure(figsize = (16,13))
    # g = sns.heatmap(corrkendall, xticklabels=corrkendall.columns.values,yticklabels=corrkendall.columns.values, cmap="RdYlGn",annot=True)
    # plt.title("Kendall rank correlation")
    # plt.show()
    
    # Get redundant variables and rank them
    def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop
    # correlation
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
    print("Top Pearsons Correlations \n", get_top_abs_correlations(df_cli2_scaled, 5))
    # Kendall
    print("Top Kendalls Correlations \n", get_top_abs_correlations(df_cli2_scaled, 5, chosen_method = 'kendall'))
    cor_support, cor_feature = cor_selector(df_cli2_scaled, df_t_scaled, 3)
    print("\n The",str(len(cor_feature)), 'Pearsons most important features:', cor_feature)   
    
    #%% grid search - define quantity of features to be used - ANOVA
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # # evaluate a given model using cross-validation
    # def evaluate_model(model, X, y):
    # 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    # 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # 	return scores
    # # define number of features to evaluate
    # num_features = [i+1 for i in range(X.shape[1])]
    # # enumerate each number of features
    # results = list()
    # for k in num_features:
    # 	# create pipeline
    # 	model = LogisticRegression(solver='liblinear')
    # 	fs = SelectKBest(score_func=f_classif, k=k)
    # 	pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
    # 	# evaluate the model
    # 	scores = evaluate_model(pipeline, X, y)
    # 	results.append(scores)
    # 	# summarize the results
    # 	#print('>%d %.3f (%.3f)' % (k, np.mean(scores), np.std(scores)))
    # dt_feat= pd.DataFrame(results, index =num_features ).T
    # plt.figure(figsize = (6,6), dpi=144)
    # bplot = sns.boxplot(data=dt_feat, width=0.5,showmeans=True).set(title = "ANOVA f-test classification accuracy for features", ylabel = 'Accuracy', xlabel = 'Number of features')
    # plt.show()    
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
    
    print("ANOVA most important features:",  X_train.iloc[:,np.argsort(fs.scores_)[-3:]].columns.tolist()[::-1])
    #%% Chi 2 select k best    
    from sklearn.feature_selection import chi2
    import sklearn
    if len(X_train.columns) > 2:
        sample = 3
    else:
        sample = 2
    bestfeatures = SelectKBest(score_func=chi2, k=sample)
    fit_chi = bestfeatures.fit(sklearn.preprocessing.MinMaxScaler().fit_transform(X_train), y_train.values.ravel())
    
    print("Chi-2 most important features:",X_train.iloc[:,np.argsort(fit_chi.scores_)[-3:]].columns.tolist()[::-1] )  #print 10 best features
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
   
    print("Mutual selection most important features:", X_train.iloc[:,np.argsort(fs.scores_)[-3:]].columns.tolist()[::-1])
    #%%
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.feature_selection import RFE
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import fbeta_score, make_scorer
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    
    # create pipeline
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
    model = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    
    # get a list of models to evaluate
    def get_models():
    	models = dict()
    	for i in range(2, 10):
    		rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
    		model = DecisionTreeClassifier()
    		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
    	return models

    # evaluate a give model using cross-validation
    def evaluate_model(model, X, y):
     	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
     	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
     	return scores
    
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
     	scores = evaluate_model(model, X, y)
     	results.append(scores)
     	names.append(name)
     	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()
    #%% random forest feature selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    rfc = RandomForestClassifier(random_state=0)
    # fit the model
    rfc.fit(X, y.values.ravel())
    # get importance
    importance = rfc.feature_importances_
    
    print("random forest classifier most important features:", X_train.iloc[:,np.argsort(importance)[-3:]].columns.tolist()[::-1])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 12), dpi=144)
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
    ax3.set_xticklabels([])
    ax4.bar(X_train.columns, fit_chi.scores_)
    ax4.set_title("Chi-2")
    
    # fig.savefig('features_rank_bar.png', bbox_inches='tight')
#%% Feature selection with all possible subsets
    if feat_sel == True:    
        from itertools import product               
        # determine the number of columns
        n_cols = X.shape[1]
        best_subset, best_score = None, 0.0
        # enumerate all combinations of input features
        for subset in product([True, False], repeat=n_cols):
        	# convert into column indexes
        	ix = [i for i, x in enumerate(subset) if x]
        	# check for now column (all False)
        	if len(ix) == 0:
        		continue
        	# select columns
        	X_new = X.values[:, ix]
        	# define model
        	model = DecisionTreeClassifier()
        	# define evaluation procedure
        	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
        	# evaluate model
        	scores = cross_val_score(model, X_new, y, scoring='f1', cv=cv, n_jobs=-1)
        	# summarize scores
        	result = np.mean(scores)
        	# report progress
        	print('>f(%s) = %f ' % (X.columns[ix], result))
        	# check if it is better than the best so far
        	if best_score is None or result >= best_score:
        		# better result
        		best_subset, best_score = X.columns[ix], result
        # report best
        print('Done!')
        print('Best subset: (%s) = %f' % (best_subset, best_score))
    
     

#%%
def failure_probability(df_cli2, df_y_f, config_hyperparameters = False, show_partial_plots = False, model_choice = 'balanced'):

    """
    This function takes as input the bias corrected EC_earth model projections,
    the months to be selected for the season (months_to_be_used = [7,8]),
    
    Parameters:
    
    The datasets for present day climate and for the 2C (optional 3C)
        
    Returns:
    
    The formatted dataframes representing EC_earth projections for PD,2C,3C
    
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    # Split data
    df_severe =pd.DataFrame( np.where(df_y_f < df_y_f.mean()-df_y_f.std(),True, False), index = df_y_f.index,columns = ['severe_loss'] ).astype(int)
    loss_intensity = df_severe
    X, y = df_cli2, loss_intensity
    #divide data train and test
    X_train, X_test, y_train, y_test = train_test_split(df_cli2, loss_intensity, test_size=0.3, random_state=0)

    #define metric to minimize false negatives
    from sklearn.metrics import fbeta_score, make_scorer
    ftwo_scorer = make_scorer(fbeta_score, beta=2)    
#%% Gridsearch CV hyperparameters # 10 min +- Use for a defined model
    if config_hyperparameters == True:
        from imblearn.ensemble import BalancedRandomForestClassifier       
        from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
        
        
        # define models and parameters
        model = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced_subsample', max_depth = 7)
        n_estimators = [100,500,600]
        max_features = [3,4,5,6]
        # define grid search
        grid = dict(n_estimators=n_estimators,max_features=max_features)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=ftwo_scorer,error_score=0)
        grid_result = grid_search.fit(X_train, y_train.values.ravel())
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        
    #%% ROC curve functions and plots
    from sklearn.metrics import confusion_matrix , plot_confusion_matrix, roc_auc_score,auc, roc_curve, precision_recall_curve, fbeta_score
    def metrics_fun(X_test, y_test, y_pred, clf, n_features = 'all'):    
        # CONFUSION MATRIX TO ASSESS METRIC
        confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 5),dpi=200)
        plot_confusion_matrix(clf, X_test, y_test,display_labels=['Negative', 'Positive'],cmap=plt.cm.Blues,ax=ax)
        plt.title(f'Confusion matrix')
        plt.show()
        
        #### ROC Curves
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        # predict probabilities
        lr_probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        roc_auc = roc_auc_score(y_test, lr_probs)

        # calculate ROC curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        # plot the roc curve for the model
        plt.figure(figsize = (5,5),dpi=200)
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
        lr_f2 = fbeta_score(y_test, y_pred, beta=2)
        lr_mcc = matthews_corrcoef(y_test, y_pred)
        
        # summarize scores
        print('SCORES: ROC-AUC=%.3f, PR-AUC=%.3f ; f1=%.3f ; f2=%.3f; MCC=%.3f' % (roc_auc, lr_auc, lr_f1, lr_f2,lr_mcc))
        
        # plot the precision-recall curves
        no_skill = len(y_test[y_test.values==1]) / len(y_test)
        plt.figure(figsize = (5,5),dpi=200)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        plt.title(f'Precision-recall - {n_features} feaures')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        return(lr_auc, lr_f1, lr_f2,lr_mcc)
        
    #%% Probabilistic failure  - weighted logistic regression
    # print("_____________________________________ \n Weighted Logistic Regression")
    # ###### all features
    # clf = LogisticRegression(random_state=0, max_iter=10e4,class_weight='balanced').fit(X_train, y_train.values.ravel())
    # importance = np.abs(clf.coef_)[0]
    # print('\n Coeficients/weights for all features: \n', importance)
    # print(f"\n All features results for {list(loss_intensity.columns.values)[0]}:")
    # print(f"{list(loss_intensity.columns.values)[0]} - training score is" , clf.score(X_train, y_train.values.ravel()))
    # print(f"{list(loss_intensity.columns.values)[0]} - test score is" , clf.score(X_test, y_test.values.ravel()))
    # y_pred = clf.predict(X_test)
    
    # ######## selecting most important features
    # number_of_features = 4
    # sel_states = SelectFromModel(LogisticRegression( random_state=0 ,max_iter=10e4),threshold=-np.inf, max_features = number_of_features)
    # sel_states.fit(X_train, y_train.values.ravel())
    # selected_feat_states = X_train.columns[(sel_states.get_support())]
    # print('\n Number of selected features: {}'.format(len(selected_feat_states)), 'which are', selected_feat_states.values)
    # #converting data to include selected features
    # X_train_selected = sel_states.transform(X_train)
    # X_test_selected = sel_states.transform(X_test)
    
    # clf_states = LogisticRegression( random_state=0,max_iter=10e4,class_weight='balanced').fit(X_train_selected, y_train.values.ravel())
    # y_pred_selected = clf_states.predict(X_test_selected)
    # score_mean_all = clf_states.score(X_test_selected, y_test.values.ravel())
    # print(f"Selected features (coef = {clf_states.coef_}) results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_states.score(X_train_selected, y_train.values.ravel()))
    # print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" ,score_mean_all )
    # scores_cv_mean_all = cross_val_score(clf_states, np.concatenate((X_train_selected, X_test_selected), axis=0), np.concatenate((y_train, y_test), axis=0).ravel(), cv=4).mean()
    # print('5 cross validation score:',scores_cv_mean_all)
    # ## Logistic
    # #all    
    # metrics_fun(X_test, y_test, y_pred, clf)
    # #selected    
    # metrics_fun(X_test_selected, y_test, y_pred_selected, clf_states, n_features=number_of_features)
    # print("_____________________________________")
        
    #%% random forest classifier
    print("_____________________________________ \n Random Forest")
    # all features
    clf_rf_all = RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1, class_weight='balanced_subsample', max_depth = 7).fit(X_train, y_train.values.ravel())
    print(f"Training score:" , clf_rf_all.score(X_train, y_train.values.ravel()),"Test score:",clf_rf_all.score(X_test, y_test.values.ravel()))

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
    # n_scores = cross_val_score(clf_rf_all, X, y.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # # report performance
    # print('CROSS VALIDATION SCORE (4 splits, 5 repeats): %.3f (%.3f) \n' % (np.mean(n_scores), np.std(n_scores)))
    
    # Predict
    y_pred = clf_rf_all.predict(X_test)
       
    # assess additional scores    
    lr_auc, lr_f1, lr_f2,lr_mcc = metrics_fun(X_test, y_test, y_pred, clf_rf_all)
    
    scoring ={'acc':'accuracy','prc':'precision','rec':'recall', 'f1': 'f1','roc_auc':'roc_auc', 'f2': ftwo_scorer}  
    n_scores_new = cross_validate(clf_rf_all, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
    print(f"CROSS VALIDATION (5 splits, 6 repeats) SCORES:",
          "Accuracy test:", round(n_scores_new['test_acc'].mean(),2),
          "Precision test:", round(n_scores_new['test_prc'].mean(),2),
          "Recall test:", round(n_scores_new['test_rec'].mean(),2),
          "ROC_AUC test:", round(n_scores_new['test_roc_auc'].mean(),2),
          "f1 test:", round(n_scores_new['test_f1'].mean(),2),
          "f2 test:", round(n_scores_new['test_f2'].mean(),2))
    
    # #select most important ones
    # sel = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=0))
    # sel.fit(X_train, y_train.values.ravel())
    # selected_feat= X_train.columns[(sel.get_support())]
    # print("\n Random Forest \n The selected features are",len(selected_feat), selected_feat.values)
    # # transform
    # X_train_selected = sel.transform(X_train)
    # X_test_selected = sel.transform(X_test)
    # # select features
    # clf_rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 6, class_weight='balanced').fit(X_train_selected, y_train.values.ravel())
    # score_mean_sel_rf = clf_rf.score(X_test_selected, y_test.values.ravel())
    # print(f"Selected features results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_rf.score(X_train_selected, y_train.values.ravel()))
    # print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" , score_mean_sel_rf )
    # y_pred_selected = clf_rf.predict(X_test_selected)
    
    # ##### plot the tree
    # from sklearn.tree import export_graphviz
    # import pydot
    # # Pull out one tree from the forest
    # tree = clf_rf.estimators_[0]
    # feature_list = list(selected_feat.values)
    # # Export the image to a dot file
    # export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # # Use dot file to create a graph
    # (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # # Write graph to a png file
    # graph.write_png('tree.png')
    
    # from PIL import Image                                                                                
    # pil_img = Image(filename='tree.png')
    # display(pil_img)

    feature_list = list(X_train.columns)
    importances = list(clf_rf_all.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[0:5]];
    
    #%% BALANCED random forest classifier - Random undersampling of the majority class in reach bootstrap sample. 
    from imblearn.ensemble import BalancedRandomForestClassifier       
    print("_____________________________________ \n Balanced Random Forest")
    
    # Model for calibration balanced random forest with parameters
    clf_brf_all = BalancedRandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1, max_depth = 6, max_features = 'sqrt')
    clf_brf_all.fit(X_train, y_train.values.ravel())
    print(f"Training score:" , clf_brf_all.score(X_train, y_train.values.ravel()),"Test score:",clf_brf_all.score(X_test, y_test.values.ravel()))
         
    # Predict
    y_pred = clf_brf_all.predict(X_test)
    
    #all    
    lr_auc, lr_f1, lr_f2,lr_mcc = metrics_fun(X_test, y_test, y_pred, clf_brf_all)

    # CROSS VALIDATION FOR MORE ROBUSTNESS
    n_scores_new = cross_validate(clf_brf_all, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
    print("CROSS VALIDATION (5 splits, 6 repeats) SCORES:",
          "Accuracy test:", round(n_scores_new['test_acc'].mean(),2),
          "Precision test:", round(n_scores_new['test_prc'].mean(),2),
          "Recall test:", round(n_scores_new['test_rec'].mean(),2),
          "ROC_AUC test:", round(n_scores_new['test_roc_auc'].mean(),2),
          "f1 test:", round(n_scores_new['test_f1'].mean(),2),
          "f2 test:", round(n_scores_new['test_f2'].mean(),2))    
    
    # #select most important ones
    # sel = SelectFromModel(BalancedRandomForestClassifier(n_estimators=1000, random_state=0))
    # sel.fit(X_train, y_train.values.ravel())
    # selected_feat= X_train.columns[(sel.get_support())]
    # print("\n Balanced Random Forest \n The selected features are",len(selected_feat), selected_feat.values)
    # # transform
    # X_train_selected = sel.transform(X_train)
    # X_test_selected = sel.transform(X_test)
    # # select features
    # clf_brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 6).fit(X_train_selected, y_train.values.ravel())
    # print(f"Selected features results: \n", f"{list(loss_intensity.columns.values)[0]} - selected training score is" , clf_brf.score(X_train_selected, y_train.values.ravel()))
    # print(f"{list(loss_intensity.columns.values)[0]} - selected test score is" , clf_brf.score(X_test_selected, y_test.values.ravel()))
    # y_pred_selected = clf_brf.predict(X_test_selected)
    
    #### plot the tree
    from IPython.display import Image 
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = clf_brf_all.estimators_[1]
    feature_list = list(X_train.columns)
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree_bal.png')
    pil_img = Image(filename='tree_bal.png')
    display(pil_img)
    
    feature_list = list(X_train.columns)
    importances = list(clf_brf_all.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[0:5]];
    features_5_top= feature_importances[0:5]
    print("_____________________________________")
    
    #%% Partial  dependence functions and plots
    if show_partial_plots == True:
        from sklearn.inspection import plot_partial_dependence
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        # X_sel = pd.DataFrame(X_train)
        # X_sel.columns = selected_feat
        ppd_feat = [x[0] for x in feature_importances][0:3]
        print(ppd_feat)
        est = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 6).fit(X_train, y_train.values.ravel())
        
        # plot_duo = plot_partial_dependence(est, X_sel, [('tmx_7_8', 'precip_7_8')]) 
       ###### plot_all = plot_partial_dependence(clf_brf_all, X_train, X_train.columns) 
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        plot_sing = plot_partial_dependence(clf_rf_all, X_train, ppd_feat, n_jobs = -1,ax=ax1) 
        fig.tight_layout()
        # plot_sing.plot()
        ax1.set_title("Partial dependence of soybean failure probability for RF")
        # fig.savefig('all_partial_plots_rbf.png', bbox_inches='tight')
        plt.show()
        
        # Second picture for BRF 
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        fig.tight_layout()
        plot_sing = plot_partial_dependence(est, X_train, ppd_feat, n_jobs = -1,ax=ax1) 

        # plot_sing.plot()
        ax1.set_title("Partial dependence of soybean failure probability for BRF")
        # fig.savefig('all_partial_plots_rbf.png', bbox_inches='tight')
        plt.show()
        ########### create the image
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), dpi=150)
        # plot_sing.plot(ax=ax1)
        # ax1.set_title("Partial dependence of soybean yield failure probability on features")
        # plot_duo.plot(ax=ax2)
        # ax2.set_title("Partial dependence of soybean yield failure probability on two features")
        # fig.savefig('partial_plot.png', bbox_inches='tight')
        # pil_img = Image(filename='partial_plot_rbf.png')
        # display(pil_img)
                     
    #%% Finalize Mahcine learning model and Return outputs function according to the type of model selected
    if model_choice == 'balanced':
        clf_brf_final = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth = 7, max_features = 'sqrt').fit(X, y.values.ravel())
    elif model_choice == 'conservative':
        clf_brf_final = RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1, 
                                               class_weight='balanced_subsample', max_depth = 7, max_features = 'sqrt').fit(X, y.values.ravel())
    
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
    # # CROSS VALIDATION FOR MORE ROBUSTNESS
    # scoring ={'acc':'accuracy','prc':'precision','f1': 'f1','roc_auc':'roc_auc', 'f2': ftwo_scorer}  
    # n_scores_final = cross_validate(clf_brf_final, X, y.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
    # print("CROSS VALIDATION (4 splits, 3 repeats) SCORES:",
    #       "Accuracy test:", round(n_scores_final['test_acc'].mean(),2),
    #       "Precision test:", round(n_scores_final['test_prc'].mean(),2),
    #       "ROC_AUC test:", round(n_scores_final['test_roc_auc'].mean(),2),
    #       "f1 test:", round(n_scores_final['test_f1'].mean(),2),
    #       "f2 test:", round(n_scores_final['test_f2'].mean(),2))  
    
    return(clf_brf_final)


