import os
import sys
from dataclasses import dataclass

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "LogisticRegression":LogisticRegression(),
                "GradientBoosting":GradientBoostingClassifier(),
                "SVM":SVC(),
                "Bagging": BaggingClassifier(),
                "GaussianNB":GaussianNB(),
                "RandomForest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier(),
                "AdaBoost":AdaBoostClassifier(),
                "LightGBM":LGBMClassifier(), 
                "CatBoost": CatBoostClassifier(),
                "ExtraTrees":ExtraTreesClassifier()
         
            }
            params = {
            'LogisticRegression': {'C': [0.1, 1, 10]},
            'SVM':  {'C':[10, 50, 100]},
            'GradientBoosting':  {# 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]},
            'Bagging': {'n_estimators': [8,16,32,64,128,256]},
            'RandomForest':  {},
            'KNN':  {'n_neighbors':[5],'n_jobs':[-1]},
            'AdaBoost':  {},
            'LightGBM':  {},
            'CatBoost':  {},
            'ExtraTrees':  {},
            'GaussianNB': {}
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)