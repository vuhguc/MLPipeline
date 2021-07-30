"""
Raw (before split) data csv file name (path)
This file is input of 01_split_data, will be split into a train data file and a test data file
"""
DATA_FILENAME = 'data/data.csv'


"""
Train data csv file name (path)
This file is produced by 01_split_data, containing data for training set
"""
TRAIN_FILENAME = 'data/train.csv'


"""
Test data csv file name (path)
This file is produced by 01_split_data, containing data for testing set
"""
TEST_FILENAME = 'data/test.csv'


"""
Ratio of test data size over full data size to be used for 01_split_data
A float between 0 and 1
Set to 0 if testing set is not used
"""
SPLIT_TEST_SIZE = 0


"""
Random state for 01_split_data
Set to an int for reproducible output or None for different output every run
Ignored if SPLIT_TEST_SIZE is set to 0
"""
SPLIT_RANDOM_STATE = 0


"""
Name of index column from data file
"""
ID_COLUMN_NAME = 'id'


"""
Columns to be used as X (input features)
Format: [(column_name, data_type)]
column_name is name of column to be included in the featureset
data_type is a string describing feature data type, can be one of the following:
    'numerical': numerical feature (float)
    'categorical': categorical feature, will be one-hot encoded
"""
X_COLUMNS = [
    ('age_over_70', 'numerical'),
    ('insurance_gov', 'numerical'),
    ('married', 'numerical'),
    ('operative_admission', 'numerical'),
    ('history_of_myocardial_infarction', 'numerical'),
    ('length_of_stay', 'numerical'),
    ('emergent_admission', 'numerical'),
    ('number_of_emergency_department_visits_within_past_6_months', 'numerical'),
    ('elective_vs_urgent_admission', 'categorical'),
    ('number_of_admissions_during_previous_year', 'numerical'),
    ('discharge_disposition', 'categorical'),
    ('number_of_admissions_in_last_6_months', 'numerical'),
    ('mdc_category_name', 'categorical')
]


"""
Column to be used as y (output value)
Format: (column_name, data_type)
column_name is name of column to be used as output value
data_type is a string describing output data type, can be one of the following:
    'numerical': numerical output (float)
    'categorical': categorical output, will be label encoded
"""
Y_COLUMN = ('unplanned_readmission', 'categorical')


"""
Preprocessor object pickle file name (path)
This file is produced by 02_preprocess_data, containing a pickle object of the created preprocessor
The preprocessor pickle object can be loaded for later steps
"""
PREPROCESSOR_OBJECT_FILENAME = 'objects/preprocessor.pickle'


"""
A boolean indicating whether 02_preprocess_data should apply scaling to X
If set to True, StandardScaler will be used to scale X. If set to False, no scaling will be applied
"""
PREPROCESS_WITH_SCALE = True


"""
Train result xlsx file name (path)
This file is produced by 03_train_model, containing a sheet describing hyperparameter tuning result on the training set, a sheet describing feature selection result on the final model (only if TRAIN_WITH_FEATURE_SELECTION is set to True), and a sheet describing nested cross validation score on the training set (only if TRAIN_WITH_NCV is set to True)
"""
TRAIN_RESULT_FILENAME = 'data/train_result.xlsx'


"""
A boolean indicating whether forward feature selection is included in 03_train_model
If set to True, forward feature selection will be used, and a sheet containing feature selection results (for the final model if nested cross validation is used) will be included in the result file. If set to False, feature selection will be skipped
"""
TRAIN_WITH_FEATURE_SELECTION = False


"""
A boolean indicating whether early stop is used for forward feature selection
If set to True, feature selection will stop as soon as the cross validation score drops. If set to False, feature selection will continue until all features are selected
Ignored if TRAIN_WITH_FEATURE_SELECTION is set to False
"""
TRAIN_WITH_FEATURE_SELECTION_EARLY_STOP = False


"""
A boolean indicating whether nested cross validation is included in 03_train_model
If set to True, nested cross validation will be used, and a sheet containing nested cross validation test scores will be included in the result file. If set to False, nested cross validation will be skipped
"""
TRAIN_WITH_NCV = False


"""
Number of folds for inner cross validation (hyperparameter tuning)
"""
TRAIN_INNER_CV = 3


"""
Number of folds for outer nested cross validation (to test trained models)
Ignored if TRAIN_WITH_NCV is set to False
"""
TRAIN_OUTER_CV = 3


"""
Models to be trained
Format: {
    'name': str,
    'model': sklearn_model,
    'param_grid': {param_name:values},
    'search_method': 'grid_search' | 'randomized_search',
    'randomized_search_n_iter': int
}
'name' is name of model
'model' is a ML model class imported from sklearn
'param_grid' is grid of all possible values for model hyperparameters. Format: {param_name:values}, where:
    param_name is name of a hyperparameter of 'model'
    values is a list of all possible values for this hyperparameter
'search_method' is method to tune hyper parameters, can be one of the following:
    'grid_search': Try all possible hyperparameter combinations and pick the best one
    'randomized_search': Generate a number of randomized hyperparameter combinations and pick the best one
'randomized_search_n_iter' is number of generated randomized parameter combinations. Ignored if 'search_method' is not 'randomized_search'
'final_model_object_filename': File name (path) of the final model object produced by 03_train_model, can be loaded for later steps
"""
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
TRAIN_MODELS = [
    # {
    #     'name': 'ada_boost',
    #     'model': AdaBoostClassifier,
    #     'param_grid': {
    #         # 'n_estimators': [25, 50, 75, 100, 125, 150, 175, 200],
    #         # 'learning_rate': [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
    #         # 'algorithm': ['SAMME', 'SAMME.R'],
    #         # 'random_state']: [0],
    #     },
    #     'search_method': 'grid_search',
    #     'final_model_object_filename': 'objects/ada_boost_classifier.pickle'
    # },
    # {
    #     'name': 'gradient_boosting',
    #     'model': GradientBoostingClassifier,
    #     'param_grid': {
    #     },
    #     'search_method': 'grid_search',
    #     'final_model_object_filename': 'objects/gradient_boosting.pickle'
    # },
   {
        'name': 'random_forest',
        'model': RandomForestClassifier,
        'param_grid': {
            'bootstrap': [True, False],
            'max_depth': [10, 50, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        },
        'search_method': 'randomized_search',
        'randomized_search_n_iter': 10,
        'final_model_object_filename': 'objects/random_forest.pickle'
    },
    {
        'name': 'logistic_regression',
        'model': LogisticRegression,
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
        },
        'search_method': 'grid_search',
        'final_model_object_filename': 'objects/logistic_regression.pickle'
    },
    # {
    #     'name': 'knn',
    #     'model': KNeighborsClassifier,
    #     'param_grid': {
    #     },
    #     'search_method': 'grid_search',
    #     'final_model_object_filename': 'objects/knn.pickle'
    # },
    # {
    #     'name': 'neural_network',
    #     'model': MLPClassifier,
    #     'param_grid': {
    #     },
    #     'search_method': 'grid_search',
    #     'final_model_object_filename': 'objects/neural_network.pickle'
    # },
    {
        'name': 'linear_svm',
        'model': LinearSVC,
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            'dual': [False, True]
        },
        'search_method': 'grid_search',
        'final_model_object_filename': 'objects/linear_svm.pickle'
    },
    {
        'name': 'decision_tree',
        'model': DecisionTreeClassifier,
        'param_grid': {
            'max_depth': [10, 50, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
        },
        'search_method': 'grid_search',
        'final_model_object_filename': 'objects/decision_tree.pickle'
    }
]


"""
Scoring metric used for hyperparameter tuning and nested cross validation test scores
Select one from https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
"""
TRAIN_SCORING_METRIC = 'roc_auc'


"""
Scoring metrics used for nested cross validation results
Select from https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
"""
NCV_SCORING_METRICS = ['roc_auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']


"""
Test result xlsx filename
This file is produced by 04_test_model, containing test scores on the testing set
"""
TEST_RESULT_FILENAME = 'data/test_result.xlsx'


"""
Scoring metrics used for test results
Select from sklearn.metrics
"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
TEST_SCORING_METRICS = [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score]
