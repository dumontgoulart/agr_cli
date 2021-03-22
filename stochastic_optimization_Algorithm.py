from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

#%% Start function

def stochastic_optimization_Algorithm(df_cli2, df_epic_det):
        
    X, y = df_cli2, df_epic_det
    y = pd.DataFrame( np.where(df_epic_det < df_epic_det.mean()-df_epic_det.std(),True, False), index = df_epic_det.index,columns = ['severe_loss'] ).astype(int)
    
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
    	X_new = X.iloc[:, ix].values
    	# define model
    	model = DecisionTreeClassifier()
    	# define evaluation procedure
    	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    	# evaluate model
    	scores = cross_val_score(model, X_new, y, scoring='f1', cv=cv, n_jobs=-1)
    	# summarize scores
    	result = np.mean(scores)
    	# report progress
    	print('>f(%s) = %f ' % (X.columns[ix].tolist(), result))
    	# check if it is better than the best so far
    	if best_score is None or result >= best_score:
    		# better result
    		best_subset, best_score = X.columns[ix].tolist(), result
    # report best
    print('Done!')
    print('Best subset: (%s) = %f' % (best_subset, best_score))
