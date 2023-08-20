import sys
import os
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(fpath)

from logger import logging
from exception import CustomException

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

class TransformPipeline:
    def __init__(self, df) -> None:
        self.predefined_mjob = ["at_home", "health", "other", "services", "teacher"]
        self.predefined_fjob = ["at_home", "health", "other", "services", "teacher"]
        self.predefined_reason = ["course", "home", "other", "reputation"]
        self.predefined_guardian = ['father', 'mother', 'other']
        self.df = df
    
    def transform_pipeline(self):
        logging.info("Transformation of new data is started")
        self.__convert_nominal_columns(self.predefined_mjob, ['Mjob'])
        self.__convert_nominal_columns(self.predefined_fjob, ['Fjob'])
        self.__convert_nominal_columns(self.predefined_reason, ['reason'])
        self.__convert_nominal_columns(self.predefined_guardian, ['guardian'])
        self.__convert_binary_columns()
        # logging.info(f'Columns type: {self.df.dtypes}')
        self.df = self.__normalize_data()
        logging.info("Transformation of new data is completed")
        return self.df
    
    def __convert_nominal_columns(self, predefined, column):
        try:
            logging.info(f"Converting {column} column data...")
            encoder = OneHotEncoder(categories=[predefined], handle_unknown='ignore')
            encoder.fit([[category] for category in predefined])
            
            # Transform the new data using the same encoder
            new_encoded_data = encoder.transform(self.df[column]).toarray()
            logging.info(f"New encoded data is {new_encoded_data}")
            
            # Create DataFrame from the encoded data
            encoded_columns = encoder.get_feature_names_out(column)
            logging.info(f"Encoded column are {encoded_columns}")
            
            new_encoded_df = pd.DataFrame(new_encoded_data, columns=encoded_columns)
            logging.info(f"New encoded dataframe is {new_encoded_df.to_string()}")
            
            self.df = pd.concat([self.df, new_encoded_df], axis=1).drop(column, axis=1)
            logging.info(f"New dataframe columns are: {self.df.columns}")
            logging.info(f"Shape: {self.df.shape}")
        except Exception as e:
            raise CustomException(e,sys)
    
    def __convert_binary_columns(self):
        logging.info("Encoding of binary columns is started....")
        binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 
                          'nursery', 'higher', 'internet', 'romantic']
        label_encoder = LabelEncoder()
        for column in binary_columns:
            logging.info(f"Binary encoding for {column}....")
            self.df[column] = label_encoder.fit_transform(self.df[column])
        logging.info(f"Dataframe after binary conversion is {self.df.to_string()}")
        
    def __normalize_data(self):
        logging.info("Started normalizing extracted features in the dataset")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.df)
        logging.info(f"Scaled data is {scaled_data}")
        scaled_df = pd.DataFrame(scaled_data, columns = self.df.columns)
        logging.info(f"Shape of dataframe is {scaled_df.shape}")
        logging.info(f"Dataframe after normalization is {scaled_df.to_string()}")
        logging.info("Completed normalizing extracted features in the dataset")
        return scaled_df
    
# if __name__ == "__main__":
#     student_data = {'school': ['GP'], 'sex': ['F'], 'age': [18], 'address': ['U'], 'famsize': ['GT3'], 'Pstatus': ['A'], 
#                     'Medu': [4], 'Fedu': [4], 'Mjob': ['at_home'], 'Fjob': ['teacher'], 'reason': ['course'], 
#                     'guardian': ['mother'], 'traveltime': [2], 'studytime': [2], 'failures': [0], 'schoolsup': ['yes'], 
#                     'famsup': ['no'], 'paid': ['no'], 'activities': ['no'], 'nursery': ['yes'], 'higher': ['yes'], 
#                     'internet': ['no'], 'romantic': ['no'], 'famrel': [4], 'freetime': [3], 'goout': [4], 'Dalc': [1], 
#                     'Walc': [1], 'health': [3], 'absences': [6], 'G1': [5], 'G2': [6]}
#     df = pd.DataFrame(student_data)
#     pipeline = TransformPipeline(df)
#     pipeline.transform_pipeline()