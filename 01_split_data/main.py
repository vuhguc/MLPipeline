import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

from config import DATA_FILENAME, TRAIN_FILENAME, TEST_FILENAME, SPLIT_TEST_SIZE, SPLIT_RANDOM_STATE

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILENAME)
    if SPLIT_TEST_SIZE <= 0:
        df_train = df
        df_test = pd.DataFrame(index=df.index, columns=df.columns)
    else:
        df_train, df_test = train_test_split(df, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    df_train.to_csv(TRAIN_FILENAME, index=False)
    df_test.to_csv(TEST_FILENAME, index=False)
