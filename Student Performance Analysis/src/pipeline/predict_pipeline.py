import sys
import os
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(fpath)

from logger import logging
from exception import CustomException
from components.data_transformation import DataTransformation
from utils.utils import Utils

class PredictPipeline:
    def __init__(self, df, sub) -> None:
        self.df = df
        self.sub = sub
    
    def predict(self):
        try:
            model_path = self.__select_sub_model(self.sub)
            utils = Utils()
            model = utils.load_model(model_path=model_path)
            logging.info(f"Successfully loading of model is completed from {model_path}")
            logging.info(f"Started predicting the performance of student based on data for {self.sub}")
            pred = model.predict(self.df)
            logging.info(f"Predicted value for new student data is {pred}")
            logging.info(f"Completed predicting the performance of student based on data for {self.sub}")
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
    def __select_sub_model(self, sub):
        logging.info(f'Fetching the model path for {sub}')
        model_path = None
        if(sub == 'maths'):
            model_path = os.path.join('../artifacts', 'maths_perf_model.pkl')
        elif(sub == 'portugese'):
            model_path = os.path.join('../artifacts', 'portugese_perf_model.pkl')
        return model_path