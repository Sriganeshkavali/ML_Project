import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def intiate_model_trainer(self, train_array, test_array):
        logging.info("Model training initiated")
        try:
            logging.info("Spliing training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]

            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'XGBRegressor': XGBRegressor(eval_metric='rmse'),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
            }

            model_report = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_name = min(model_report, key=lambda x: model_report[x]['rmse_test'])
            best_model_score = model_report[best_model_name]
            logging.info(f"Best model found: {best_model_name} with RMSE: {best_model_score['rmse_test']}")
            
            best_model = models[best_model_name]
            if best_model_score['r2_score_test'] < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score['r2_score_test']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score of the best model on test data: {r2_square}")

            return best_model_name, r2_square
        except Exception as e:
            raise CustomException(e, sys) from e
