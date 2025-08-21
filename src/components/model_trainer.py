import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()
      
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train,y_train = train_array[:, :-1], train_array[:, -1]
            X_test,y_test = test_array[:, :-1], test_array[:, -1]
            
            models = {
              "Linear Regression": LinearRegression(),
               "K-NeighborsRegressor": KNeighborsRegressor(),
               "DecisionTreeRegressor": DecisionTreeRegressor(),
               "RandomForestRegressor": RandomForestRegressor(),
               "XGBRegressor": XGBRegressor(),
               "AdaBoostRegressor": AdaBoostRegressor()
            }
            params={
              "DecisionTreeRegressor":{
                'criterion':['friedman_mse', 'squared_error', 'absolute_error'],
                'splitter':['best', 'random'],
                'max_features':['sqrt', 'log2']
              },
              "RandomForestRegressor":{
                'criterion':['friedman_mse', 'squared_error', 'absolute_error'],
                'n_estimators':[10,32,64,50, 100],
                'max_features':['sqrt', 'log2'],
                'min_samples_split':[2, 5, 10]
              },
              "Gradient Boosting":{
                'loss':['ls', 'lad', 'huber', 'quantile'],
                'learning_rate':[0.01, 0.1, 0.2],
                'n_estimators':[100, 200, 300],
                'max_depth':[3, 5, 7],
                'min_samples_split':[2, 5, 10],
                'max_features':['sqrt', 'log2']
              },
              "K-NeighborsRegressor": {
                'n_neighbors':[3, 5, 7, 9, 11],
                'weights':['uniform', 'distance'],
                'algorithm':['ball_tree', 'kd_tree']
              },
              "XGBRegressor": {
                'n_estimators':[100, 200, 300],
                'learning_rate':[0.01, 0.1, 0.2],
                'max_depth':[3, 5, 7],
                'min_child_weight':[1, 3, 5],
                'subsample':[0.6, 0.8, 1.0],
                'colsample_bytree':[0.6, 0.8, 1.0]
              },
              'Linear Regression': {},
              
              'AdaBoostRegressor': {
                'n_estimators':[50, 100, 200],
                'learning_rate':[0.01, 0.1, 0.5],
                'loss':['linear', 'square', 'exponential']
              }

            }


            model_report = evaluate_model(X_train, X_test, y_train, y_test, models,param=params)


            best_model_score = max(sorted(model_report.values()))
            
            best_model_name =list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
              file_path=self.model_trainer_config.trained_model_file_path,
              object=best_model
            )
            
            predicted = best_model.predict(X_test)  
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
          raise CustomException(e, sys) 
