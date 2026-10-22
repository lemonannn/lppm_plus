# Creates a Simple Analysis

# First, modify your config to determine the desired predictors for this analysis (example in config.json)
# Basically init() -> get_predictors() -> get_lppm_scores() -> get_model_metrics()
# Maybe I'll make it all happen at init at some point but it's convienient that it's seperate right now

# I also want to tune the hyperparameters, probably just for random forest though realistically

# Also, there's almost certainly a better way to handle stuff+ model being included
# Don't even know if it currently works, but I'll check soon

import pandas as pd
from fip_predictor import FIP_Predictor
import helpy
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt

class Analysis:
    
    # Useful Variables:
    # self.results_df contains all collected data for this analysis (available after running get_lppm_scores())
    # self.predictors contains all predictors used in this analysis (available after running get_predictors())
    
    def __init__(self, analysis_df, training_range, testing_range, config_file="config.json"):
        ## Plus data only available for 2020+
        
        self.analysis_df = analysis_df.copy()
        self.training_range = training_range
        self.testing_range = testing_range
        
        self.plus_df = analysis_df[analysis_df['Season'] > 2020].copy()
            
        self.config = helpy.load_config(config_file)
        
    def full_analysis(self):
        
           self.get_predictors()
           self.get_lppm_scores()
           self.get_model_metrics()
           
             

    def get_predictors(self):
        # Creates and trains FIP- predictors as per the given config file
        
        predictors = self.get_predictor_list()

        results = []

        for name, predictor in predictors.items():

            result = predictor.create_models(name, model_config=self.config["Feature Sets"][name]["models"], 
                                             tuning_config=self.config["tuning"])

            result.insert(0, "Feature Set", name)

            results.append(result)

        self.predictor_results = pd.concat(results, ignore_index=True)
        self.predictors = predictors
        
        
    def get_predictor_list(self):
        # Returns the list of configured predictors   
        
        predictors = {}
        
        for name, settings in self.config["Feature Sets"].items():
            
            if not settings['enabled']:
                continue
            
            predictors[name] = FIP_Predictor(
                self.plus_df if settings['data'] == 'plus' else self.analysis_df,
                training_range=self.training_range,
                testing_range=self.testing_range,
                **self.config['Predictors'][name]
            )
            
        return predictors
    
    def get_baseline_pred(self):
        # Returns the FIP- prediction using just previous year FIP-
        
        baseline = FIP_Predictor(
            self.analysis_df,
            training_range=self.training_range,
            testing_range=self.testing_range,
            **self.config['Predictors']['Baseline']
        )
        
        baseline.create_gam('Baseline')
        
        return baseline.pred_gam
        
    
    def get_lppm_scores(self):
        # Calculates scores for selected columns
        
        # Create test datasets for score creation
        analysis_df_test = self.analysis_df[
            self.analysis_df["Season"].between(*self.testing_range)
        ].copy()
        
        baseline_col = self.get_baseline_pred()

        # Build prediction columns only for selected predictors in config.json
        for key, predictor in self.predictors.items():

            predictions = predictor.predictions
            
            for prediction in predictions:
            
                model_name = prediction['model_name']
                y_pred = prediction['pred_y']
                
                col = f"pred_{model_name}"
                analysis_df_test[col] = y_pred
                
                
                # Score each selected model against the baseline prediction
                analysis_df_test[f"{model_name}_score"] = baseline_col - y_pred

        self.results_df = analysis_df_test
                
                
    def get_model_metrics(self, df=None, display_figure=False, baseline=True):
        # Returns selected model ROC-AUC and PR-AUC, and creates lift curve figure

        metrics = []
        
        plt.figure(figsize=(10, 6))
        
        models = []
        
        if df is None:
            df = self.results_df.copy()
            
        if baseline:
            baseline_df = helpy.create_baseline_breakout_model(self.analysis_df, self.training_range, self.testing_range)
            df['score_fip'] = baseline_df['score_fip']
            models.append(('Baseline Model', 'score_fip'))
            
            metrics.append((get_metrics(df, 'Baseline Model', 'score_fip')))
            
        for key, predictor in self.predictors.items():
        
            predictions = predictor.predictions
            
            for prediction in predictions:
                
                model_name = prediction['model_name']
            
                score_col = f"{model_name}_score"
        
                metrics.append(get_metrics(df, model_name, score_col))
                
                models.append((model_name, score_col)) 
                
        if display_figure:
            self.create_lift_plot(models, df)
                
        self.breakout_metrics = pd.DataFrame(metrics)
        
        return (
            self.breakout_metrics
            .sort_values("PR-AUC", ascending=False)
            .reset_index(drop=True)
        )
                
    def create_lift_plot(self, models, df):
        # creates lift curve figure
        # models should contain a list of (display name, column name, dataframe type)

        plt.figure(figsize=(10, 6))

        for display_name, score_col in models:
            
            data = df[["led_to_breakout", score_col]]
            
            y = data["led_to_breakout"]
            score = data[score_col]
            
            percentiles, lift = helpy.lift_curve(y, -score)

            plt.plot(
                percentiles * 100,
                lift,
                label=display_name
            )
            
        plt.axhline(1, linestyle="--", label="Random")
                
        plt.xlim(0, 50)
        plt.xticks([0, 5, 10, 20, 30, 40, 50])

        plt.xlabel("Top X% of pitchers ranked by model")
        plt.ylabel("Lift")
        plt.title("Breakout Lift by Model Ranking")

        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
        
        
def get_metrics(df, model_name, col_name):
    data = df[["led_to_breakout", col_name]]
                    
    y = data["led_to_breakout"]
    score = data[col_name]
    
    # creates base metrics table
    roc_auc = roc_auc_score(y, score)
    pr_auc = average_precision_score(y, score)
    
    return {
            "Model": model_name,
            "N": len(data),
            "Breakouts": int(y.sum()),
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }