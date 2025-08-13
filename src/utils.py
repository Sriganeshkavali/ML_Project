import os 
import sys
import pandas as pd
import numpy as np
import dill
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