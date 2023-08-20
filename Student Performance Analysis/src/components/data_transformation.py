import os
import sys
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(fpath)
from logger import logging
from exception import CustomException

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

class DataTransformation():
    def __init__(self, df) -> None:
        self.df = df
        
    def convert_categorical_to_numerical(self):
        '''
            Converts the categorical columns to numerical columns
        '''
        # new_df = self.df.copy()
        logging.info("Started converting categorical columns to numerical columns.")
        categorical_features = self.df.select_dtypes(include=['object', 'bool']).columns
        logging.info(f'Categorical Features are: {categorical_features}')
        
        # differentiating binary columns and nominal columns
        binary_columns = []
        nominal_columns = []
        
        for column in categorical_features:
            unique_value = self.df[column].nunique()
            if unique_value == 2:
                binary_columns.append(column)
            elif unique_value > 2:
                nominal_columns.append(column)

        logging.info(f'Binary columns are: {binary_columns}')
        logging.info(f'Nominal columns are: {nominal_columns}')
        
        def convert_binary_to_numerical():
            label_encoder = LabelEncoder()
            for column in binary_columns:
                self.df[column] = label_encoder.fit_transform(self.df[column])
        
        def convert_nominal_to_numerical():
            one_hot_encoder = OneHotEncoder(sparse=False)
            encoded_data_mjob = pd.DataFrame(one_hot_encoder.fit_transform(self.df[nominal_columns]),
                                         columns=one_hot_encoder.get_feature_names_out())
            self.df = pd.concat([self.df, encoded_data_mjob], axis=1).drop(nominal_columns, axis=1)
            
        convert_binary_to_numerical()
        convert_nominal_to_numerical()
        logging.info(f"Columns: {self.df.columns}")
        logging.info(f"Shape after converting categorical column to numerical columns: {self.df.shape}")
        logging.info("Completed converting categorical columns to numerical columns.")
        # return self.df
    
    def basic_data_cleaning(self):
        '''
            Will do basic data cleaning on the dataset
        '''
        logging.info("Started cleaning the dataset")
        # drops the single value columns
        def remove_single_value_column():
            counts = self.df.nunique()
            to_del = [i for i,v in enumerate(counts) if v == 1]
            if len(to_del) > 0:
                self.df.drop(to_del, axis=1, inplace=True)
            logging.info(f'Shape after removing single value column: {self.df.shape}')
        
        # drop duplicate value rows
        def remove_duplicate_value_rows():
            dups = self.df.duplicated()
            if dups.any():
                logging.info(f"Duplicates: {self.df[dups]}")
                self.df.drop_duplicates(inplace=True)
            logging.info(f'Shape after removing duplicate value: {self.df.shape}')

        logging.info(f'Shape before doing basic data cleaning: {self.df.shape}')
        remove_single_value_column()
        remove_duplicate_value_rows()
        logging.info("Completed cleaning the dataset")
        # return self.df
    
    def fill_missing_value(self):
        '''
            Fill missing values in the dataset
        '''
        logging.info("Started filling missing values in the dataset")
        missing_value_columns = self.df.columns[self.df.isna().any()]
        logging.info(f'Missing value columns: {missing_value_columns}')
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for column in missing_value_columns:
            imputer.fit(self.df[column].values.reshape(-1, 1))
            self.df[column] = imputer.fit_transform(self.df[column].values.reshape(-1, 1))
        logging.info(f'Missing value columns after transform using SimpleImputer: {self.df.columns[self.df.isna().any()]}')
        logging.info("Completed filling missing values in the dataset")
        # return self.df
    
    def extract_features(self, criteria, k):
        '''
            Extract the features from the dataset
        '''
        logging.info("Started extracting features from the dataset")
        X = self.df.drop(['G3'], axis=1).values
        y = self.df['G3'].values
        feature_selection = SelectKBest(score_func=criteria, k=k)
        feature_selection.fit_transform(X, y)
        cols_idxs = feature_selection.get_support(indices=True)
        logging.info("Completed extraction of features from the dataset")
        self.df= self.df.iloc[:, cols_idxs]
        logging.info(f"Extracted columns name: {self.df.columns}")
        
    def normalize_data(self):
        '''
            Normalize the dataset
        '''
        logging.info("Started normalizing extracted features in the dataset")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.df)
        scaled_df = pd.DataFrame(scaled_data, columns = self.df.columns)
        logging.info(f"Shape of dataframe is {scaled_df.shape}")
        logging.info("Completed normalizing extracted features in the dataset")
        return scaled_df