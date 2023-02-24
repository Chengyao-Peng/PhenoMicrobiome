from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score,explained_variance_score
import os
import pandas as pd
import joblib
import optuna
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_regression, RFECV, SelectFromModel, SelectKBest, f_regression
from sklearn.neighbors import KNeighborsRegressor
import math
import operator
import numpy as np

def objective(trial):
    global X_train_, y_train
    X_train_ = pd.read_csv('./data/X_genus_shared_filteredLow_relative_genus_train.csv', index_col=0)
    y_train = pd.read_csv('./data/y_genus_shared_filteredLow_relative_genus_train.csv', index_col=0).to_numpy().flatten()

    # different models 
#     taxonomic_level = trial.suggest_categorical('taxonomic level', ['Pylum','Class','Order','Family','Genus', 'Species'])
    # feature_selection = trial.suggest_categorical('Feature selection methods', ["coda4microbiome", "Selbal"])
    feature_selection = trial.suggest_categorical('Feature seletion',['F-statistic', 'Mutual information', 'Lasso', 'Tree-based', 'None'])

    if feature_selection == 'F-statistic':
        k = trial.suggest_int('Selected k', 10, len(X_train_.columns))
        
        def getTopFeatures(train_x, train_y, n_features):
            f_val, p_val = f_regression(train_x,train_y)
            f_val_dict = {}
            p_val_dict = {}
            for i in range(len(f_val)):
                if math.isnan(f_val[i]):
                    f_val[i] = 0.0
                f_val_dict[i] = f_val[i]
                if math.isnan(p_val[i]):
                    p_val[i] = 0.0
                p_val_dict[i] = p_val[i]

            sorted_f = sorted(f_val_dict.items(), key=operator.itemgetter(1),reverse=True)
            sorted_p = sorted(p_val_dict.items(), key=operator.itemgetter(1),reverse=True)

            feature_indexs = []
            for i in range(0,n_features):
                feature_indexs.append(sorted_f[i][0])

            return feature_indexs

        selected_features = getTopFeatures(X_train_, y_train, 20)
        X_train = X_train_[X_train_.columns[selected_features]]
        
        
    elif feature_selection == 'Mutual information':
        k = trial.suggest_int('Selected k', 10, len(X_train_.columns))
        selector = SelectKBest(mutual_info_regression, k=k).fit(X_train_, y_train)
        mask = selector.get_support()
        X_train = X_train_[X_train_.columns[mask]]

    elif feature_selection == 'Lasso':
        selector= LassoCV().fit(X_train_, y_train)
        importance = np.abs(selector.coef_)
        sfm = SelectFromModel(selector, threshold=0.00001)
        sfm.fit(X_train_, y_train)
        X_train = sfm.transform(X_train_)

    elif feature_selection == 'Tree-based':
        selector = SelectFromModel(RandomForestRegressor(n_estimators = 100),threshold=0.00001)
        selector.fit(X_train_, y_train)
        mask = selector.get_support()
        X_train = X_train_[X_train_.columns[mask]]
    else:
        X_train = X_train_

    
    param = { 
            'alpha': trial.suggest_loguniform("alpha", 1e-5, 1e5),
            'l1_ratio': trial.suggest_float("l1_ratio", 0, 1, step=0.01)
        }

    regressor_obj = ElasticNet(**param)
    explained_variances = cross_val_score(regressor_obj, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=2), scoring="explained_variance")
    
    return explained_variances.mean()




if __name__ == "__main__":
	#optimizing study
	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=50,show_progress_bar=True)

	#learn
	joblib.dump(study, './results/elastic_net_batchCorrected.pkl')

