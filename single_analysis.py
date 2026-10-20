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
            
            

    def get_predictors(self):
        
        predictors = self.get_predictor_list()

        results = []

        for name, predictor in predictors.items():

            result = predictor.create_models(self.config["Feature Sets"][name]["models"])

            result.insert(0, "Feature Set", name)

            results.append(result)

        self.predictor_results = pd.concat(results, ignore_index=True)
        self.predictors = predictors
        
        
    def get_predictor_list(self):   
        
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
        
        baseline = FIP_Predictor(
            self.analysis_df,
            training_range=self.training_range,
            testing_range=self.testing_range,
            **self.config['Predictors']['Baseline']
        )
        
        baseline.create_gam()
        
        return baseline.pred_gam
        
    
    def get_lppm_scores(self):
        # Calculates scores for selected columns
        
        # Create test datasets for score creation
        analysis_df_test = self.analysis_df[
            self.analysis_df["Season"].between(*self.testing_range)
        ].copy()

        plus_df_test = self.plus_df[
            self.plus_df["Season"].between(*self.testing_range)
        ].copy()


        # Build prediction columns only for selected predictors in config.json
        plus_cols = ["PlayerName", "Season"]
        score_cols = []

        for name, settings in self.config['Feature Sets'].items():
            # ex: Name: Residuals 
            # settings: enabled: True, data: plus, models: {rf: true, en: true}
            
            if not settings['enabled']:
                continue

            target_df = (
                analysis_df_test
                if settings["data"] == "analysis"
                else plus_df_test
            )

            for model, activated in settings['models'].items():
                # ex: model: rf, activated: True
                
                if activated == False:
                    continue
                
                col = f"pred_{name}_{model}"
                target_df[col] = getattr(self.predictors[name], f"pred_{model}")
                
                if settings['data'] == 'plus':
                    plus_cols.append(col)
                
                score_cols.append(col)

        # Merge selected plus-model predictions into the main DataFrame
        analysis_df_test = analysis_df_test.merge(
            plus_df_test[plus_cols],
            on=["PlayerName", "Season"],
            how="left",
            validate="one_to_one",
        )

        # Score each selected model against the baseline prediction
        baseline_col = self.get_baseline_pred()
            
        for name, settings in self.config['Feature Sets'].items():
            # ex: Name: Residuals 
            # settings: enabled: True, data: plus, models: {rf: true, en: true}
            
            if not settings['enabled']:
                continue
            
            for model, activated in settings['models'].items():
                # ex: model: rf, activated: True
                
                if not activated:
                    continue
                
                analysis_df_test[f"{name}_{model}_score"] = baseline_col - analysis_df_test[f"pred_{name}_{model}"]

        self.results_df = analysis_df_test
                
                
    def get_model_metrics(self):
        # Returns selected model ROC-AUC and PR-AUC, and creates lift curve figure
        
        results_df = self.results_df.copy()
    
        self.results_plus_df = results_df[results_df['Season'] >= 2021]

        metrics = []
        
        plt.figure(figsize=(10, 6))
        
        models = []

        for name, settings in self.config['Feature Sets'].items():
            # ex: Name: Residuals 
            # settings: enabled: True, data: plus, models: {rf: true, en: true}
            
            if not settings['enabled']:
                continue
            
            df = self.results_plus_df if settings['data'] == 'plus' else results_df
            
            for model, activated in settings['models'].items():
                # ex: model: rf, activated: True
                
                if not activated:
                    continue
                
                score_col = f"{name}_{model}_score"
                
                data = df[["led_to_breakout", score_col]]
                
                y = data["led_to_breakout"]
                score = data[score_col]
                
                # creates base metrics table
                roc_auc = roc_auc_score(y, score)
                pr_auc = average_precision_score(y, score)
                
                metrics.append({
                    "Model": f"{name} {model}",
                    "N": len(data),
                    "Breakouts": int(y.sum()),
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc,
                })
                
                models.append((f"{name} {model}", score_col, settings['data']))
                
        self.create_lift_plot(models)
        
        self.breakout_metrics = pd.DataFrame(metrics)
        
        return (
            self.breakout_metrics
            .sort_values("PR-AUC", ascending=False)
            .reset_index(drop=True)
        )
                
    def create_lift_plot(self, models):
        # creates lift curve figure
        # models should contain a list of (display name, column name, dataframe type)

        plt.figure(figsize=(10, 6))

        for display_name, score_col, data_type in models:
            
            df = self.results_plus_df if data_type == 'plus' else self.results_df
            
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