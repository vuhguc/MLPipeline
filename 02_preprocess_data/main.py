import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

from config import TRAIN_FILENAME, ID_COLUMN_NAME, X_COLUMNS, Y_COLUMN, PREPROCESSOR_OBJECT_FILENAME, PREPROCESS_WITH_SCALE
from ml_utils.preprocessing import Preprocessor

import pandas as pd
import pickle

if __name__ == '__main__':
    df_train = pd.read_csv(TRAIN_FILENAME).set_index(ID_COLUMN_NAME)
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train, X_COLUMNS, Y_COLUMN, PREPROCESS_WITH_SCALE)
    with open(PREPROCESSOR_OBJECT_FILENAME, 'wb') as object_file:
        pickle.dump(preprocessor, object_file)
