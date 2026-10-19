import pandas as pd
from fip_predictor import FIP_Predictor
import helpy
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import json

class Analysis:
    
    def __init__(self, analysis_df, training_range, testing_range, include_plus=False):
        ## Plus data only available for 2020+
        
        self.analysis_df = analysis_df.copy()
        self.training_range = training_range
        self.testing_range = testing_range
        self.include_plus = include_plus
        
        if include_plus:
            self.plus_df = analysis_df[analysis_df['Season'] > 2020].copy()
        else:
            self.plus_df = pd.DataFrame()
            

    def get_predictors(self, *, residuals=True, usage=False, movement=False, baseline=True,
        no_fip=False, velo_fip=False, stuff_plus=False, command_plus=False, pitch_present=True):
        
        self.residuals = residuals
        self.usage = usage
        self.movement = movement
        self.baseline = baseline
        self.no_fip = no_fip
        self.velo_fip = velo_fip
        self.stuff_plus = stuff_plus
        self.command_plus = command_plus
        self.pitch_present = pitch_present
        
        predictors = self.get_predictor_list()

        results = []

        for name, predictor in predictors.items():

            result = predictor.create_all_models()

            result.insert(0, "Feature Set", name)

            results.append(result)

        self.predictor_results = pd.concat(results, ignore_index=True)
        self.predictors = predictors
        
        
    def get_predictor_list(self):   
        groups = {
            "Residuals": self.residuals,
            "Usage": self.usage,
            "Regular Movement": self.movement,
            "Baseline": self.baseline,
            "No FIP": self.no_fip,
            "Velo + FIP": self.velo_fip,
            "Stuff+": self.stuff_plus,
            "Stuff+ and Command+": self.command_plus,
            "Command+": self.command_plus,
        }

        predictors = {}

        for name, enabled in groups.items():
            if not enabled:
                continue

            is_plus = name in {
                "Stuff+",
                "Stuff+ and Command+",
                "Command+",
            }

            if is_plus and not self.include_plus:
                continue

            predictors[name] = FIP_Predictor(
                self.plus_df if is_plus else self.analysis_df,
                training_range=self.training_range,
                testing_range=self.testing_range,
                **helpy.PREDICTOR_CONFIGS[name],
            )
            
        return predictors
    
    def get_lppm_scores(self):

        analysis_df_test = self.analysis_df[
            self.analysis_df["Season"].between(*self.testing_range)
        ].copy()

        if self.include_plus:
            plus_df_test = self.plus_df[
                self.plus_df["Season"].between(*self.testing_range)
            ].copy()

        configs = {
            "Residuals": {
                "df": "analysis",
                "residuals_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Velo + FIP": {
                "df": "analysis",
                "velo_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Regular Movement": {
                "df": "analysis",
                "movement_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Usage": {
                "df": "analysis",
                "usage_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Baseline": {
                "df": "analysis",
                "baseline_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Stuff+": {
                "df": "plus",
                "stuff_plus_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Command+": {
                "df": "plus",
                "command_plus_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
            "Stuff+ and Command+": {
                "df": "plus",
                "all_plus_pred_fip-": {
                    "rf": "pred_rf",
                    "en": "pred_elastic",
                    "xgb": "pred_xgb",
                    "gam": "pred_gam"
                },
            },
        }

        # Build prediction columns only for selected predictors
        plus_cols = ["PlayerName", "Season"]

        for name, config in configs.items():
            if name not in self.predictors:
                continue

            target_df = (
                analysis_df_test
                if config["df"] == "analysis"
                else plus_df_test
            )

            for prefix, models in config.items():
                if prefix == "df":
                    continue

                for model, attribute in models.items():
                    col = f"{prefix}{model}"
                    
                    target_df[col] = getattr(self.predictors[name], attribute)

                    if config["df"] == "plus":
                        plus_cols.append(col)

        # Merge selected plus-model predictions into the main DataFrame
        if len(plus_cols) > 2:
            analysis_df_test = analysis_df_test.merge(
                plus_df_test[plus_cols],
                on=["PlayerName", "Season"],
                how="left",
                validate="one_to_one",
            )

        # Score each selected model against the baseline prediction
        baseline_col = "baseline_pred_fip-gam"

        score_map = {
            "residuals_pred_fip-rf": "score_residuals_rf",
            "residuals_pred_fip-en": "score_residuals_en",
            "residuals_pred_fip-xgb": "score_residuals_xgb",
            "residuals_pred_fip-gam": "score_residuals_gam",
            "velo_pred_fip-rf": "score_velo_rf",
            "velo_pred_fip-en": "score_velo_en",
            "velo_pred_fip-xgb": "score_velo_xgb",
            "velo_pred_fip-gam": "score_velo_gam",
            "movement_pred_fip-rf": "score_movement_rf",
            "movement_pred_fip-en": "score_movement_en",
            "movement_pred_fip-xgb": "score_movement_xgb",
            "movement_pred_fip-gam": "score_movement_gam",
            "usage_pred_fip-rf": "score_usage_rf",
            "usage_pred_fip-en": "score_usage_en",
            "usage_pred_fip-xgb": "score_usage_xgb",
            "usage_pred_fip-gam": "score_usage_gam",
            "stuff_plus_pred_fip-en": "score_stuff_en",
            "stuff_plus_pred_fip-rf": "score_stuff_rf",
            "stuff_plus_pred_fip-xgb": "score_stuff_xgb",
            "stuff_plus_pred_fip-gam": "score_stuff_gam",
            "command_plus_pred_fip-rf": "score_command_rf",
            "command_plus_pred_fip-xgb": "score_command_xgb",
            "command_plus_pred_fip-en": "score_command_en",
            "command_plus_pred_fip-gam": "score_command_gam",
            "all_plus_pred_fip-rf": "score_all_plus_rf",
            "all_plus_pred_fip-xgb": "score_all_plus_xgb",
            "all_plus_pred_fip-en": "score_all_plus_en",
            "all_plus_pred_fip-gam": "score_all_plus_gam",
        }

        if baseline_col in analysis_df_test.columns:
            for prediction_col, score_col in score_map.items():
                if prediction_col in analysis_df_test.columns:
                    analysis_df_test[score_col] = (
                        analysis_df_test[baseline_col]
                        - analysis_df_test[prediction_col]
                    )

        self.results_df = analysis_df_test
                
                
    def get_model_metrics(self):
        
        results_df = self.results_df.copy()
        
        if self.include_plus:
            results_plus_df = results_df.dropna(
                subset=["score_stuff_en"]
            ).copy()

        # configue this to customize results
        score_cols = self.get_score_cols()

        metrics = []

        for predictor_name, models in score_cols.items():
            if predictor_name not in self.predictors:
                continue

            is_plus = predictor_name in {
                "Stuff+",
                "Command+",
                "Stuff+ and Command+",
            }

            df = results_plus_df if is_plus else results_df

            for model_name, col in models.items():
                if col not in df.columns:
                    continue

                data = df[["led_to_breakout", col]].dropna()

                y = data["led_to_breakout"]
                score = data[col]

                # ROC-AUC requires both classes to be present
                if y.nunique() < 2:
                    roc_auc = float("nan")
                else:
                    roc_auc = roc_auc_score(y, score)

                pr_auc = (
                    average_precision_score(y, score)
                    if len(y) > 0 and y.sum() > 0
                    else float("nan")
                )

                metrics.append({
                    "Model": model_name,
                    "N": len(data),
                    "Breakouts": int(y.sum()),
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc,
                })
                
        # Plot lift curves for selected models
        plt.figure(figsize=(10, 6))

        for predictor_name, models in score_cols.items():
            if predictor_name not in self.predictors:
                continue

            is_plus = predictor_name in {
                "Stuff+",
                "Command+",
                "Stuff+ and Command+",
            }

            df = results_plus_df if is_plus else results_df

            for name, col in models.items():
                if col not in df.columns:
                    continue

                data = df[["led_to_breakout", col]].dropna()

                if data.empty or data["led_to_breakout"].nunique() < 2:
                    continue

                percentiles, lift = helpy.lift_curve(
                    data["led_to_breakout"],
                    -data[col]
                )

                plt.plot(
                    percentiles * 100,
                    lift,
                    label=name
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
        
        self.breakout_metrics = pd.DataFrame(metrics)

        return (
            self.breakout_metrics
            .sort_values("PR-AUC", ascending=False)
            .reset_index(drop=True)
        )
        
    def get_score_cols(self):
        
        config = self.load_config()

        return {
            group: {
                name: settings["column"]
                for name, settings in models.items()
                if settings["enabled"]
            }
            for group, models in config["score_cols"].items()
            if any(settings["enabled"] for settings in models.values())
        }
    
    def load_config(self, config_path="config.json"):
        with open(config_path, "r") as f:
            return json.load(f)