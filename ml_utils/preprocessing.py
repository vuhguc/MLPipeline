import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(cur_dir)
sys.path.append(cur_dir)

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

class Preprocessor:
    
    def __init__(self):
        self.X_numerical_column_names = None
        self.X_categorical_column_names = None
        self.X_categorical_encoder = None
        self.X_scaler = None
        self.y_column_name = None
        self.y_encoder = None
        
    def fit_transform(self, df, X_columns, y_column=None, with_scale=False):
        self.X_numerical_column_names = None
        self.X_categorical_column_names = None
        self.X_categorical_encoder = None
        self.X_scaler = None
        self.y_column_name = None
        self.y_encoder = None

        self.X_numerical_column_names = []
        self.X_categorical_column_names = []
        for column_name, data_type in X_columns:
            if data_type == 'numerical':
                self.X_numerical_column_names.append(column_name)
            elif data_type == 'categorical':
                self.X_categorical_column_names.append(column_name)
            else:
                raise Exception('Unknown data type {} for column {} of X.'.format(data_type, column_name))
        X = df[self.X_numerical_column_names].astype(float).fillna(0)
        if len(self.X_categorical_column_names) > 0:
            self.X_categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            X_encoded = self.X_categorical_encoder.fit_transform(df[self.X_categorical_column_names].astype(str)).astype(float)
            encoded_column_names = self.X_categorical_encoder.get_feature_names(self.X_categorical_column_names)
            X[encoded_column_names] = X_encoded
        if with_scale:
            self.X_scaler = StandardScaler()
            X[:] = self.X_scaler.fit_transform(X)
        y_column_name, y_data_type = y_column
        if y_column_name is not None:
            self.y_column_name = y_column_name
            y = df[y_column_name].copy()
            if y_data_type == 'numerical':
                y = y.astype(float)
            elif y_data_type == 'categorical':
                self.y_encoder = LabelEncoder()
                y[:] = self.y_encoder.fit_transform(y.astype(str))
            else:
                raise Exception('Unknown data type {} for column {} of y.'.format(y_data_type, y_column_name))
        return (X, y)

    def transform(self, df):
        X = df[self.X_numerical_column_names].astype(float).fillna(0)
        y = None
        if len(self.X_categorical_column_names) > 0:
            X_encoded = self.X_categorical_encoder.transform(df[self.X_categorical_column_names].astype(str)).astype(float)
            encoded_column_names = self.X_categorical_encoder.get_feature_names(self.X_categorical_column_names)
            X[encoded_column_names] = X_encoded
        if self.X_scaler is not None:
            X[:] = self.X_scaler.transform(X)
        if self.y_column_name is not None and self.y_column_name in df:
            y = df[self.y_column_name].copy()
            if self.y_encoder is None:
                y = y.astype(float)
            else:
                y[:] = self.y_encoder.transform(y.astype(str))
        return (X, y)
