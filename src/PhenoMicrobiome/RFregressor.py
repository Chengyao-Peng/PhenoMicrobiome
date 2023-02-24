import os
import pandas as pd 
import sklearn.model_selection 
from sklearn.model_selection import train_test_split 
import optuna
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def split_dataset(X, y):
    '''This function split X and y in to trainng and test set by 80:20.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#     X_train, X_val, y_train, y_val 
#     = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    
#     return X_train, X_val, X_test, y_train, y_val , y_test
    return X_train, X_test, y_train, y_test

def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
    bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
    max_depth = trial.suggest_int('max_depth', 1, 10000)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 10000)
    n_estimators =  trial.suggest_int('n_estimators', 30, 1000)
    
    regr = RandomForestRegressor(bootstrap = bootstrap, criterion = criterion,
                                 max_depth = max_depth, max_features = max_features,
                                 max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,n_jobs=3)
    
    
    #regr.fit(X_train, y_train)
    #y_pred = regr.predict(X_val)
    #return r2_score(y_val, y_pred)
    
    score = cross_val_score(regr, X_train, y_train, cv=10, scoring="r2")
    r2_mean = score.mean()

    return r2_mean

if __name__ == "__main__":
	# Prepocessing
	all_table_path = os.path.join('..', 'all_taxa_emission.txt' )
	all_table_df = pd.read_csv(all_table_path, sep='\t', header=0, index_col=0)
	all_table_df = all_table_df.dropna(axis='columns')
	all_table_df.iloc[:, :-1] = (all_table_df.iloc[:, :-1] - all_table_df.iloc[:, :-1].mean())/all_table_df.iloc[:, :-1].std()
	all_table_df = all_table_df.dropna(axis=1)
	features = all_table_df.iloc[:, :-1].columns.tolist() 
	target = all_table_df.iloc[:, -1:].columns.tolist()

	# Create X, y
	X = all_table_df[features]
	y = all_table_df[target]

	# Split X, y
	X_train, X_test, y_train, y_test = split_dataset(X, y)

	#optimizing study
	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=100,show_progress_bar=True)

	#Create an instance with tuned hyperparameters
	optimised_rf = RandomForestRegressor(bootstrap = study.best_params['bootstrap'], criterion = study.best_params['criterion'],
                                     max_depth = study.best_params['max_depth'], max_features = study.best_params['max_features'],
                                     max_leaf_nodes = study.best_params['max_leaf_nodes'],n_estimators = study.best_params['n_estimators'],
                                     n_jobs=3)
	#learn
	optimised_rf.fit(X_train ,y_train)
	joblib.dump(study, 'RFregressor_100trials.pkl')
