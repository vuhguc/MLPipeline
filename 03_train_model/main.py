import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

from config import PREPROCESS_RESULT_FILENAME, ID_COLUMN_NAME, TRAIN_MODELS, TRAIN_SCORING_METRIC, NCV_SCORING_METRICS, TRAIN_INNER_CV, TRAIN_WITH_FEATURE_SELECTION, TRAIN_WITH_FEATURE_SELECTION_EARLY_STOP, TRAIN_WITH_NCV, TRAIN_OUTER_CV, TRAIN_RESULT_FILENAME
from utils.feature_selection import ForwardFeatureSelector

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
import pickle

if __name__ == '__main__':
    X_train = pd.read_excel(PREPROCESS_RESULT_FILENAME, sheet_name='X_train', engine='openpyxl').set_index(ID_COLUMN_NAME)
    y_train = pd.read_excel(PREPROCESS_RESULT_FILENAME, sheet_name='y_train', engine='openpyxl').set_index(ID_COLUMN_NAME).iloc[:, 0]
    with pd.ExcelWriter(TRAIN_RESULT_FILENAME) as writer:
        for model in TRAIN_MODELS:

            if model['search_method'] == 'grid_search':
                param_tuner = GridSearchCV(model['model'](), param_grid=model['param_grid'], scoring=TRAIN_SCORING_METRIC, cv=TRAIN_INNER_CV, n_jobs=-1)
            elif model['search_method'] == 'randomized_search':
                param_tuner = RandomizedSearchCV(model['model'](), param_distributions=model['param_grid'], n_iter=model['randomized_search_n_iter'], scoring=TRAIN_SCORING_METRIC, cv=TRAIN_INNER_CV, n_jobs=-1)
            else:
                raise Exception('Unknown parameter tuning search method {}.'.format(model['search_method']))

            if TRAIN_WITH_FEATURE_SELECTION:
                feature_selector = ForwardFeatureSelector(clone(param_tuner), early_stop=TRAIN_WITH_FEATURE_SELECTION_EARLY_STOP, verbose=True)
                pipeline = Pipeline([
                    ('feature_selector', feature_selector),
                    ('param_tuner', param_tuner),
                ])
            else:
                pipeline = Pipeline([
                    ('param_tuner', param_tuner)
                ])

            pipeline.fit(X_train, y_train)

            if TRAIN_WITH_FEATURE_SELECTION:
                fs_results = pd.DataFrame(pipeline['feature_selector'].fs_results_)
            cv_results = pd.DataFrame(pipeline['param_tuner'].cv_results_)

            if TRAIN_WITH_NCV:
                scores = cross_validate(pipeline, X_train, y_train, scoring=NCV_SCORING_METRICS, cv=TRAIN_OUTER_CV)
                ncv_results_dict = {'fold':list(range(TRAIN_OUTER_CV))+['avg']}
                for score in scores:
                    ncv_results_dict[score] = list(scores[score]) + [scores[score].mean()]
                ncv_results = pd.DataFrame(ncv_results_dict).set_index('fold')

            if TRAIN_WITH_FEATURE_SELECTION:
                fs_results.to_excel(writer, sheet_name=model['name']+'_fs_results')
            cv_results.to_excel(writer, sheet_name=model['name']+'_cv_results')
            if TRAIN_WITH_NCV:
                ncv_results.to_excel(writer, sheet_name=model['name']+'_ncv_results')

            with open(model['final_model_object_filename'], 'wb') as object_file:
                pickle.dump(pipeline, object_file)
