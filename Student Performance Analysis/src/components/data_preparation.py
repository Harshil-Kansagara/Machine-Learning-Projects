import sys
import os
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(fpath)

from dataclasses import dataclass
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from utils.utils import Utils

import pandas as pd

@dataclass
class DataPreparationConfig:
    stud_maths_transformed_csv_path: str = os.path.join("../artifacts", "maths_performance_transformed.csv")
    stud_portugese_transformed_csv_path: str = os.path.join("../artifacts", "portugese_performance_transformed.csv")

class DataPreparation(DataTransformation):
    def __init__(self, maths_perf_csv_path, portugese_perf_csv_path) -> None:
        self.preparation_config = DataPreparationConfig()
        self.maths_perf_csv_path = maths_perf_csv_path
        self.portugese_perf_csv_path = portugese_perf_csv_path
        maths_perf_df = self.read_perf_csv(maths_perf_csv_path)
        super().__init__(maths_perf_df)
        self.data_transformation_methods('maths')
        port_perf_df = self.read_perf_csv(portugese_perf_csv_path)
        super().__init__(port_perf_df)
        self.data_transformation_methods('portugese')
    
    def read_perf_csv(self, perf_file_path):
        '''
            Read student performance file
        '''
        logging.info(f"Reading student {os.path.basename(perf_file_path)} performance csv file..")
        perf_df = pd.read_csv(perf_file_path, sep=";")
        return perf_df
    
    def data_transformation_methods(self, sub):
        logging.info(f"Started transforming dataset for {sub} performance file")
        try:
            self.convert_categorical_to_numerical()
            self.basic_data_cleaning()
            self.fill_missing_value()
            # self.extract_features(f_regression, 30)
            transformed_df = self.normalize_data()
            logging.info(f"Completed transforming dataset for {sub} performance file")
            file_path = (lambda sub: self.preparation_config.stud_maths_transformed_csv_path if sub == 'maths' else self.preparation_config.stud_portugese_transformed_csv_path)(sub)
            self.save_generated_preprocess_df_to_csv_file(file_path, transformed_df)
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_generated_preprocess_df_to_csv_file(self, file_path, df):
        '''
            Save generated preprocessed dataframe to csv file at particular location
        '''
        utils = Utils()
        utils.save_csv_object_file(file_path, df)
