import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

import pandas as pd

class ForwardFeatureSelector:
    
    EARLY_STOP_MAX_DECREASE = 0.001
    
    def __init__(self, param_tuner, early_stop=False, verbose=False):
        self.param_tuner = param_tuner
        self.early_stop = early_stop
        self.verbose = verbose
        self.best_features_ = None
        self.best_score_ = None
        self.fs_results_ = None

    def fit_transform(self, X, y):
        features = []
        feature_set = set()
        self.best_features_ = None
        self.best_score_ = None
        self.fs_results_ = {'features':[], 'score':[], 'rank':[]}
        while len(feature_set) < len(X.columns):
            best_feature = None
            best_score = None
            for feature in X.columns:
                if feature not in feature_set:
                    features.append(feature)
                    X_sub = X[features]
                    self.param_tuner.fit(X_sub, y)
                    score = self.param_tuner.best_score_
                    if best_score is None or score > best_score:
                        best_feature = feature
                        best_score = score
                    features.pop()
            features.append(best_feature)
            feature_set.add(best_feature)
            if self.best_score_ is None or best_score > self.best_score_:
                self.best_features_ = list(features)
                self.best_score_ = best_score
            self.fs_results_['features'].append(str(features))
            self.fs_results_['score'].append(best_score)
            if self.verbose:
                print(type(self.param_tuner.estimator).__name__, features, best_score)
            if self.early_stop and best_score + self.EARLY_STOP_MAX_DECREASE < self.best_score_:
                break
        self.fs_results_['rank'] = pd.Series(self.fs_results_['score']).rank(method='min', ascending=False).tolist()
        return X[self.best_features_]

    def fit(self, X, y):
        self.fit_tranform(X, y)
        return self

    def transform(self, X):
        return X[self.best_features_]
