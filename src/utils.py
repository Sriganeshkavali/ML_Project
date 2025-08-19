import os 
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import logging

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    import pickle
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Saving object at {file_path}")
        # Use dill to handle more complex objects if necessary
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple regression models and return their performance metrics.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            r2_train = r2_score(y_train, y_pred_train)

            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)
            model_report[model_name] = {'rmse_test': rmse_test, 'r2_score_test': r2_test}
            logging.info(f"{model_name} - RMSE: {rmse_test}, R2 Score: {r2_test}")

        return model_report
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path):
    """
    Load an object from a file using pickle.
    """
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e