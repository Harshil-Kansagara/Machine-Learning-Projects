import sys
import os
fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(fpath)

from logger import logging
from exception import CustomException
import pandas as pd

class StudentData:
    def __init__(self, school: str, sex: str, age: int, address: str, famsize: str, pstatus: str, medu: int, fedu: int, 
                 mjob: str, fjob: str, reason: str, guardian: str, traveltime: int, studytime:int, failures: int, 
                 schoolsup:int, famsup:int, paid:int, activities:int, nursery:int, higher:int, internet: int, romantic: int, 
                 famrel:int, freetime:int, goout:int, dalc:int, walc: int, health: int, absences, g1: int, g2: int) -> None:
        self.student_input_data_dict = {
            "school": [school],
            "sex" : [sex],
            "age" : [age],
            "address" : [address],
            "famsize" : [famsize],
            "Pstatus" : [pstatus],
            "Medu" : [medu],
            "Fedu" : [fedu],
            "Mjob" : [mjob],
            "Fjob" : [fjob],
            "reason" : [reason],
            "guardian" : [guardian],
            "traveltime" : [traveltime],
            "studytime" : [studytime],
            "failures" : [failures],
            "schoolsup" : ['yes'] if schoolsup==0 else ['no'],
            "famsup" : ['yes'] if famsup==0 else ['no'],
            "paid" : ['yes'] if paid==0 else ['no'],
            "activities" : ['yes'] if activities==0 else ['no'],
            "nursery" : ['yes'] if nursery==0 else ['no'],
            "higher" : ['yes'] if higher==0 else ['no'],
            "internet" : ['yes'] if internet==0 else ['no'],
            "romantic" : ['yes'] if romantic==0 else ['no'],
            "famrel" : [famrel],
            "freetime" : [freetime],
            "goout" : [goout],
            "Dalc" : [dalc],
            "Walc" : [walc],
            "health" : [health],
            "absences" : [absences],
            "G1": [g1],
            "G2": [g2]
        }
        logging.info(f"Student Data is: {self.student_input_data_dict}")
    
    def convert_raw_data_to_data_frame(self):
        try:
            logging.info("Converting raw data to data frame..")
            return pd.DataFrame(self.student_input_data_dict)
            # logging.info(f"New data frame: {new_data}")
            # new_encoded_data = pd.get_dummies(new_data, columns=['Mjob'])
            # aligned_new_encoded_data, aligned_original_data = new_encoded_data.align(new_data, fill_value=0)
            # combined_data = pd.concat([aligned_original_data, aligned_new_encoded_data])
            # logging.info(f"Encoded data is: {combined_data}")
            # return combined_data
        except Exception as e:
            raise CustomException(e, sys)