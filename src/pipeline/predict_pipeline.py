import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
                 
        Area: float,
        Perimeter:float,
        Compactness:float,
        Length_of_kernel:float,
        Width_of_kernel:float,
        Asymmetry_coefficient:float,
        Length_of_kernel_groove:float):

        self.Area = Area

        self.Perimeter = Perimeter

        self.Compactness = Compactness

        self.Length_of_kernel = Length_of_kernel

        self.Width_of_kernel = Width_of_kernel

        self.Asymmetry_coefficient = Asymmetry_coefficient

        self. Length_of_kernel_groove =  Length_of_kernel_groove

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Area": [self.Area],
                "Perimeter": [self.Perimeter],
                "Compactness": [self.Compactness],
                "Length_of_kernel": [self.Length_of_kernel],
                "Width_of_kernel": [self.Width_of_kernel],
                "Asymmetry_coefficient": [self.Asymmetry_coefficient],
                " Length_of_kernel_groove": [self. Length_of_kernel_groove],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)