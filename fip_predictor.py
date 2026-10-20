# Single FIP- Predictor, with XGB, EN, RF, and GAM models for the selected variables included

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

class FIP_Predictor:
    
    def __init__(self, analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False, stuff_plus=False, command_plus=False, pitching_plus=False, testing_range=(2020, 2025), training_range=(2000, 2020)):
        
        self.X, self.X_train, self.X_test, self.y_train, self.y_test = helpy.get_train_test_split(analysis_df, velo, fip, movement, usage, movement_residuals, stuff_plus, command_plus, pitching_plus, testing_range=testing_range, training_range=training_range)
        
        
    def create_elastic_net(self):
    
        elastic = make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42
            )
        )

        elastic.fit(self.X_train, self.y_train)

        pred_en = elastic.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(elastic)
        
        self.elastic = elastic
        self.pred_en = pred_en
        
        return mae, rmse, r2

    def create_xgboost(self):

        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        xgb.fit(self.X_train, self.y_train)

        pred_xgb = xgb.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(xgb)
        
        self.xgb = xgb
        self.pred_xgb = pred_xgb
        
        return mae, rmse, r2
        
    def create_random_forest(self):
    
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )

        rf.fit(self.X_train, self.y_train)
        
        pred_rf = rf.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(rf)
        
        self.rf = rf
        self.pred_rf = pred_rf

        return mae, rmse, r2
    
    def create_gam(self):

        gam_terms = terms.s(0)

        for i in range(1, self.X_train.shape[1]):
            gam_terms += terms.s(i)

        gam = LinearGAM(gam_terms)
        gam.fit(self.X_train, self.y_train)
        
        pred_gam = gam.predict(self.X_test)
        
        mae, rmse, r2 = self.eval_model(gam)
        
        self.gam = gam
        self.pred_gam = pred_gam
        
        return mae, rmse, r2
    
    def eval_model(self, model):
        
        pred = model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))
        r2 = r2_score(self.y_test, pred)
        
        return mae, rmse, r2
        
    def create_models(self, model_config, tuning_mode=False):

        if tuning_mode:
            return

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

            mae, rmse, r2 = create_model()

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        return pd.DataFrame(results)
    
        