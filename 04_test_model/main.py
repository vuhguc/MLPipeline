import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

from config import PREPROCESSOR_OBJECT_FILENAME, TRAIN_MODELS, TEST_FILENAME, ID_COLUMN_NAME, TEST_SCORING_METRICS, TEST_RESULT_FILENAME

import pickle
import pandas as pd

if __name__ == '__main__':
    with open(PREPROCESSOR_OBJECT_FILENAME, 'rb') as object_file:
        preprocessor = pickle.load(object_file)
    with pd.ExcelWriter(TEST_RESULT_FILENAME) as writer:
        for model in TRAIN_MODELS:
            with open(model['final_model_object_filename'], 'rb') as object_file:
                final_model = pickle.load(object_file)
            df_test = pd.read_csv(TEST_FILENAME, dtype=str).set_index(ID_COLUMN_NAME)
            X_test, y_test = preprocessor.transform(df_test)
            y_pred = final_model.predict(X_test)
            prediction_result = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred}, index=df_test.index)
            metric_names = [metric.__name__ for metric in TEST_SCORING_METRICS]
            scores = [metric(y_test, y_pred) for metric in TEST_SCORING_METRICS]
            avg_score = sum(scores) / len(scores)
            evaluation_result = pd.DataFrame({'metric':metric_names+['avg'], 'score':scores+[avg_score]}).set_index('metric')
            prediction_result.to_excel(writer, sheet_name=model['name']+'_prediction_result')
            evaluation_result.to_excel(writer, sheet_name=model['name']+'_evaluation_result')
