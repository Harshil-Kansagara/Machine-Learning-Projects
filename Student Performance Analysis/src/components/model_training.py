import os
import sys
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(fpath)

from dataclasses import dataclass
from exception import CustomException
from logger import logging
from components.data_preparation import DataPreparationConfig
from utils.utils import Utils

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainingConfig(DataPreparationConfig):
    trained_model_maths_performance: str = os.path.join('../artifacts', 'maths_perf_model.pkl')
    trained_model_portugese_performance: str = os.path.join('../artifacts', 'portugese_perf_model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.training_config = ModelTrainingConfig()
    
    def split_data(self, file_path):
        try:
            logging.info(f"Splitting the {os.path.basename(file_path)} performance data into train and test data")
            df = pd.read_csv(file_path, sep=";")
            X, y = df.drop(['G3'], axis=1).values, df['G3'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
            logging.info(f"Splitting completed for {os.path.basename(file_path)} performance data into train and test data")
            logging.info(f"Shape of train X data: {X_train.shape}")
            logging.info(f"Shape of test X data: {X_test.shape}")
            logging.info(f"Shape of train Y data: {y_train.shape}")
            logging.info(f"Shape of test Y data: {y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, X_train, X_test, y_train, y_test, sub):
        try:
            logging.info("Initiate model training..")
            models = {
                "Linear Regression": LinearRegression(),
                # "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "XGBoost Regressor": XGBRegressor(),
                "Cat Boosting Algorithm": CatBoostRegressor(verbose=False),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "XGBoost Regressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boosting Algorithm": {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Ada Boost Regressor": {
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Lasso": {
                    'alpha': [1e-15, 1e-10, 1e-08, 0.001, 0.01, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]
                },
                "Ridge": {
                    'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                }
            }
            
            utils = Utils()
            model_report: dict = utils.evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                       models=models, model_params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best model found on both training and testing dataset")
            
            file_path = (lambda sub: self.training_config.trained_model_maths_performance if sub == 'maths' else self.training_config.trained_model_portugese_performance)(sub)
            utils.save_object(file_path=file_path, obj=best_model)
            
            pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, pred)
            logging.info("Completed model training")
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)