import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

class ModelOptimizer:
    
    def __init__(model, search_method, param_grid, scoring, cv, randomized_search_n_iter=None, df=None, X_columns=None, y_column=None, with_scale=False):
        if search_method == 'grid_search':
            self.param_tuner = GridSearchCV(model(), param_grid=param_grid, scoring=scoring, cv=cv)
        elif search_method == 'randomized_search':
            self.param_tuner = RandomizedSearchCV(model(), param_distributions=param_grid, n_iter=randomized_search_n_iter, scoring=scoring, cv=cv)
        self.df = df
        self.X_columns = X_columns
        self.y_column = y_column
        self.with_scale = with_sacle
        self.preprocessor = None

    def fit(X, y):
        
