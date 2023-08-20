import sys
import os
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(fpath)

import pandas as pd
from dataclasses import dataclass
from exception import CustomException
from logger import logging

from components.data_preparation import DataPreparation
from components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    maths_raw_data_path: str = os.path.join("../artifacts", "student-math.csv")
    portugese_raw_data_path: str = os.path.join("../artifacts", "student-por.csv")
    data_folder_path: str = os.path.join("../artifacts", "")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Started creating folder for storing the artifacts")
            os.makedirs(os.path.dirname(self.ingestion_config.data_folder_path), exist_ok=True)
            logging.info("Completed creating folder for storing the artifacts")
            self.read_maths_performance_csv()
            self.read_portugese_performance_csv()
            logging.info("Ingestion of data completed.")
            return (self.ingestion_config.maths_raw_data_path,
                    self.ingestion_config.portugese_raw_data_path)
        except Exception as e:
            raise CustomException(e, sys)

    def read_maths_performance_csv(self):
        logging.info("Reading student performance csv")
        df = pd.read_csv('../notebook/datasets/student-mat.csv', sep=";")
        df.to_csv(self.ingestion_config.maths_raw_data_path, sep=";", index=False, header=True)
        logging.info("Completed reading student performance csv")
        
    def read_portugese_performance_csv(self):
        logging.info("Reading protugese performance csv")
        df = pd.read_csv('../notebook/datasets/student-por.csv', sep=";")
        df.to_csv(self.ingestion_config.portugese_raw_data_path, sep=";", index=False, header=True)
        logging.info("Completed reading portugese performance csv")

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    maths_perf_csv_path, portugese_perf_csv_path = data_ingestion.initiate_data_ingestion()
    data_preparation = DataPreparation(maths_perf_csv_path=maths_perf_csv_path, portugese_perf_csv_path=portugese_perf_csv_path)
    
    model_training = ModelTraining()
    X_train, X_test, y_train, y_test = model_training.split_data(data_preparation.preparation_config.stud_maths_transformed_csv_path)
    print(f"Accuracy for Maths Performance dataset: {model_training.initiate_model_training(X_train, X_test, y_train, y_test, 'maths')}")
    X_train, X_test, y_train, y_test = model_training.split_data(data_preparation.preparation_config.stud_portugese_transformed_csv_path)
    print(f"Accuracy for Portugese Performance dataset: {model_training.initiate_model_training(X_train, X_test, y_train, y_test, 'portugese')}")