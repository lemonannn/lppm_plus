# Single FIP- Predictor, with XGB, EN, RF, and GAM models for the selected variables included
# All the functions here are pretty self evident

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s, terms
import helpy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

class FIP_Predictor:
    
    def __init__(self, analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False, stuff_plus=False, command_plus=False, pitching_plus=False, testing_range=(2020, 2025), training_range=(2000, 2020)):
        
        self.X, self.X_train, self.X_test, self.y_train, self.y_test = helpy.get_train_test_split(analysis_df, velo, fip, movement, usage, movement_residuals, stuff_plus, command_plus, pitching_plus, testing_range=testing_range, training_range=training_range)
        
        self.predictions = []
        
        
    def create_elastic_net(self, predictor_name, params=None):
        
        if params == None:
            params = default_params["en"]
    
        elastic = make_pipeline(
            StandardScaler(),
            ElasticNet(
                **params
            )
        )

        elastic.fit(self.X_train, self.y_train)

        pred_en = elastic.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(elastic)
        
        self.elastic = elastic
        self.pred_en = pred_en
        
        self.predictions.append({'model_name': f'{predictor_name}_en', 'model': elastic, 'pred_y': pred_en, 'hyperparameters': params})
        
        return mae, rmse, r2

    def create_xgboost(self, predictor_name, params=None):
        
        if params == None: 
            params = default_params["xgb"]

        xgb = XGBRegressor(
            **params
        )

        xgb.fit(self.X_train, self.y_train)

        pred_xgb = xgb.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(xgb)
        
        self.xgb = xgb
        self.pred_xgb = pred_xgb
        
        self.predictions.append({'model_name': f'{predictor_name}_xgb', 'model': xgb, 'pred_y': pred_xgb, 'hyperparameters': params})
        
        return mae, rmse, r2
        
    def create_random_forest(self, predictor_name, params=None):
        
        if params == None:
            params = default_params['rf']
    
        rf = RandomForestRegressor(**params)

        rf.fit(self.X_train, self.y_train)
        
        pred_rf = rf.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(rf)
        
        self.rf = rf
        self.pred_rf = pred_rf
        
        self.predictions.append({'model_name': f'{predictor_name}_rf', 'model': rf, 'pred_y': pred_rf, 'hyperparameters': params})

        return mae, rmse, r2
    
    def create_gam(self, predictor_name, params=None):

        gam_terms = terms.s(0)

        for i in range(1, self.X_train.shape[1]):
            gam_terms += terms.s(i)

        gam = LinearGAM(gam_terms)
        gam.fit(self.X_train, self.y_train)
        
        pred_gam = gam.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(gam)
        
        self.gam = gam
        self.pred_gam = pred_gam
        
        self.predictions.append({'model_name': f'{predictor_name}_gam', 'model': gam, 'pred_y': pred_gam, 'hyperparameters': params})
        
        return mae, rmse, r2
    
    def eval_model(self, model):
        
        pred = model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))
        r2 = r2_score(self.y_test, pred)
        
        return mae, rmse, r2
        
    def create_models(self, predictor_name, model_config, tuning_config):

        model_methods = {
            "en": ("Elastic Net", self.create_elastic_net),
            "xgb": ("XGBoost", self.create_xgboost),
            "rf": ("Random Forest", self.create_random_forest),
            "gam": ("GAM", self.create_gam)
        }

        results = []

        for key, (name, create_model) in model_methods.items():

            if not model_config.get(key, False):
                continue
            
            params_list = [None]

            if tuning_config.get("enabled", False):
                tuning_model_config = tuning_config["models"][key]

                params_list = list(ParameterSampler(
                    tuning_model_config["params"],
                    n_iter=tuning_config["n_iter"],
                    random_state=tuning_config.get("random_state", 42)
                ))
                
            i = 0

            for params in params_list:
                mae, rmse, r2 = create_model(predictor_name + str(i), params=params)
                i = i + 1

                results.append({
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

        return pd.DataFrame(results)
    
    
default_params = {
    "en": {
        "alpha": 0.01,
        "l1_ratio": 1.0,
        "max_iter": 100000,
        "tol": 1e-4,
        "random_state": 42,
    },

    "rf": {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "random_state": 42,
        "n_jobs": -1,
    },

    "xgb": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1,
        "reg_lambda": 10,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    },
}
        