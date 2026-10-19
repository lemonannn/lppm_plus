## Helper Functions

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# list of pitch types per fangraphs data
pitch_types = ["FA", "FC", "SI", "CH", "SL", "CU"]

# list of preset predictor model configurations
# will still only apply if added to the predictor list in single_analysis.py and activated
PREDICTOR_CONFIGS = {
    "Residuals": {
        "usage": False,
        "movement": False,
        "movement_residuals": True,
    },
    "Usage": {
        "movement": False,
        "movement_residuals": False,
    },
    "Regular Movement": {
        "usage": False,
        "movement": True,
        "movement_residuals": False,
    },
    "Baseline": {
        "velo": False,
        "movement": False,
        "usage": False,
        "movement_residuals": False,
    },
    "No FIP": {
        "fip": False,
        "movement": False,
        "movement_residuals": True,
        "usage": False,
    },
    "Velo + FIP": {
        "movement": False,
        "usage": False,
        "movement_residuals": False,
    },
    "Stuff+": {
        "velo": False,
        "movement": False,
        "usage": False,
        "movement_residuals": False,
        "stuff_plus": True,
    },
    "Stuff+ and Command+": {
        "velo": False,
        "movement": False,
        "usage": False,
        "pitching_plus": True,
        "command_plus": True,
    },
    "Command+": {
        "movement": False,
        "movement_residuals": True,
        "usage": False,
        "command_plus": True,
    },
}

# Starting Season for the data
STARTING_YEAR = pd.read_csv('arsenal_data.csv')['Season'].min()

# basic info, result stats
# pitch velocity info
# pitch usage info
# pitch movement info (horizontal)
# pitch movement info (vertical)
# Stuff+ and Command+
ANALYSIS_COLUMNS = [
    "PlayerName", "Season", "FIP-", "ERA-",
    *[f"piv{p}" for p in ["FA", "FC", "FS", "SI", "CH", "SL", "CU", "CS", "KN"]],
    *[f"pi{p}%" for p in ["FA", "FC", "FS", "SI", "CH", "SL", "CU", "CS", "KN"]],
    *[f"pi{p}-{axis}" for axis in ["X", "Z"]
        for p in ["FA", "FC", "FS", "SI", "CH", "SL", "CU", "CS", "KN"]],
    *[f"pb_{metric}" for metric in ["command", "stuff", "overall"]],
    *[f"pb_{prefix}_{p}" for prefix in ["c", "s", "o"]
        for p in ["CH", "CU", "FF", "SI", "SL", "KC", "FC", "FS"]],
]


def is_breakout(analysis_df, FIP_percent_decrease_threshold, FIP_overall_threshold, breakout_year_buffer=3):
    # Returns dataset with led_to_breakout column
    
    analysis_df = analysis_df.copy()
    analysis_df['led_to_breakout'] = False

    baseline_fip = analysis_df['FIP-']

    valid_baseline = baseline_fip.notna() & (baseline_fip > 0)

    for i in range(1, breakout_year_buffer + 1):
        future_fip = analysis_df[
            'next' + '_next' * (i - 1) + '_FIP-'
        ]

        mask = (
            valid_baseline
            & future_fip.notna()
            & (future_fip <= FIP_overall_threshold)
            & (
                1 - future_fip / baseline_fip
                >= FIP_percent_decrease_threshold
            )
        )

        analysis_df.loc[mask, 'led_to_breakout'] = True

    return analysis_df

def get_year_df(year):
    # Returns dataset with one year of basic pitch data
    
    df = pd.read_csv('arsenal_data.csv')

    filtered_df = (
        df[ANALYSIS_COLUMNS]
        .copy()
        .sort_values(["PlayerName", "Season"])
        .reset_index(drop=True)
    )
    
    filtered_df = filtered_df[filtered_df['Season'] == year]
    
    filtered_df = predict_residuals(filtered_df)
    
    filtered_df = handle_na(filtered_df)
    
    return filtered_df
    
def get_analysis_df(FIP_decrease_threshold, FIP_overall_threshold, breakout_year_buffer):
    # Return complete analysis df
    # Includes, most importantly, the led_to_breakout column
    
    df = pd.read_csv('arsenal_data.csv')
    
    filtered_df = (
        df[ANALYSIS_COLUMNS]
        .copy()
        .sort_values(["PlayerName", "Season"])
        .reset_index(drop=True)
    )
    
    grouped = filtered_df.groupby("PlayerName", sort=False)

    # Future seasons
    for i in range(1, breakout_year_buffer+1):
        prefix = "next" + "_next" * (i - 1)

        filtered_df[f"{prefix}_Season"] = grouped["Season"].shift(-i)
        filtered_df[f"{prefix}_FIP-"] = grouped["FIP-"].shift(-i)

    # Classify breakouts using the starting season as the baseline.
    filtered_df = is_breakout(
        filtered_df,
        FIP_decrease_threshold,
        FIP_overall_threshold,
        breakout_year_buffer,
    )

    # Keep analysis years and rows with known baseline and next-season FIP-.
    analysis_df = filtered_df.loc[
        (filtered_df["Season"] >= 2007)
        & filtered_df["FIP-"].notna()
        & filtered_df["next_FIP-"].notna()
    ].copy()
    
    analysis_df = predict_residuals(analysis_df)
    
    analysis_df = handle_na(analysis_df)

    return analysis_df

def handle_na(df):
    # Handles NA values for pitches that aren't present
    # Creates a present column for each pitch
    
    analysis_df = df.copy()
    
    for p in pitch_types:
        usage_col = f"pi{p}%"

        # Pitch is present if usage is greater than zero
        present = analysis_df[usage_col].fillna(0) > 0

        # Add indicator
        analysis_df[f"{p}_present"] = present.astype(int)

        # Zero out characteristics for pitches not thrown
        pitch_cols = [
            f"piv{p}",
            f"pi{p}-X",
            f"pi{p}-Z",
            f"{p}_X_residual",
            f"{p}_Z_residual",
            f"pb_s_{p}",
            f"pb_c_{p}",
            f"pb_o_{p}",
        ]

        pitch_cols = [col for col in pitch_cols if col in analysis_df.columns]

        analysis_df.loc[~present, pitch_cols] = 0

        # Missing usage means the pitch is not thrown
        analysis_df[usage_col] = analysis_df[usage_col].fillna(0)
            
    return analysis_df

def get_train_test_split(analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False, stuff_plus=False, command_plus=False, pitching_plus=False, pitch_present=True, testing_range=(2020, 2025), training_range=(2000, 2020)):
    # Returns training and testing datasets for model creation

    feature_groups = {
        "velo": [f"piv{p}" for p in pitch_types],
        "fip": ["FIP-"],
        "movement": [
            *[f"pi{p}-X" for p in pitch_types],
            *[f"pi{p}-Z" for p in pitch_types],
        ],
        "usage": [f"pi{p}%" for p in pitch_types],
        "movement_residuals": [
            *[f"{p}_X_residual" for p in pitch_types],
            *[f"{p}_Z_residual" for p in pitch_types],
        ],
        "stuff_plus": [f"pb_s_{p}" for p in pitch_types],
        "command_plus": [f"pb_c_{p}" for p in pitch_types],
        "pitching_plus": [f"pb_o_{p}" for p in pitch_types],
        "pitch_present": [f"{p}_present" for p in pitch_types],
    }

    options = {
        "velo": velo,
        "fip": fip,
        "movement": movement,
        "usage": usage,
        "movement_residuals": movement_residuals,
        "stuff_plus": stuff_plus,
        "command_plus": command_plus,
        "pitching_plus": pitching_plus,
        "pitch_present": pitch_present,
    }

    feature_cols = [
        col
        for group, enabled in options.items()
        if enabled
        for col in feature_groups[group]
    ]

    if not feature_cols:
        raise ValueError("At least one feature group must be enabled.")

    missing_cols = set(feature_cols + ["Season", "next_FIP-"]) - set(
        analysis_df.columns
    )
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    X = analysis_df[feature_cols].copy()
    
    y = analysis_df["next_FIP-"]

    train_mask = analysis_df["Season"].between(*training_range)
    test_mask = analysis_df["Season"].between(*testing_range)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return X, X_train, X_test, y_train, y_test



def predict_residuals(df):
    # returns df with movement residuals
    
    analysis_df = df.copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

        pitch_types = ["FA", "FC", "FS", "SI", "CH", "SL", "CU", "CS", "KN"]

        for p in pitch_types:

            # Initialize columns
            analysis_df[f"expected_{p}_Z"] = np.nan
            analysis_df[f"expected_{p}_X"] = np.nan

            # ----------------
            # Z-break model
            # ----------------
            z_mask = (
                analysis_df[f"piv{p}"].notna() &
                analysis_df[f"pi{p}-Z"].notna()
            )

            if z_mask.sum() > 1:
                z_model = LinearRegression()

                z_model.fit(
                    analysis_df.loc[z_mask, [f"piv{p}"]],
                    analysis_df.loc[z_mask, f"pi{p}-Z"]
                )

                analysis_df.loc[z_mask, f"expected_{p}_Z"] = (
                    z_model.predict(
                        analysis_df.loc[z_mask, [f"piv{p}"]]
                    )
                )

            # ----------------
            # X-break model
            # ----------------
            x_mask = (
                analysis_df[f"piv{p}"].notna() &
                analysis_df[f"pi{p}-X"].notna()
            )

            if x_mask.sum() > 1:
                x_model = LinearRegression()

                x_model.fit(
                    analysis_df.loc[x_mask, [f"piv{p}"]],
                    analysis_df.loc[x_mask, f"pi{p}-X"]
                )

                analysis_df.loc[x_mask, f"expected_{p}_X"] = (
                    x_model.predict(
                        analysis_df.loc[x_mask, [f"piv{p}"]]
                    )
                )
                
        for p in pitch_types:

            analysis_df[f"{p}_Z_residual"] = (
                analysis_df[f"pi{p}-Z"]
                - analysis_df[f"expected_{p}_Z"]
            )

            analysis_df[f"{p}_X_residual"] = (
                analysis_df[f"pi{p}-X"]
                - analysis_df[f"expected_{p}_X"]
            )
            
    return analysis_df

def breakout_rate_at_percentage(df, score_col, breakout_col="led_to_breakout", percentage=0.10):
    # returns rolling breakout rate at specific percentage
    
    n = max(1, int(len(df) * percentage))

    top = df.nlargest(n, score_col)

    return {
        "N": n,
        "Breakouts": int(top[breakout_col].sum()),
        "Breakout Rate": top[breakout_col].mean(),
        "Overall Breakout Rate": df[breakout_col].mean(),
        "Lift": top[breakout_col].mean() / df[breakout_col].mean()
    }
    
def breakout_rate_suite(df, score_col, breakout_col="led_to_breakout"):
    # Returns rolling breakout rate at 5%, 10%, 25%, 50%
    
    five = breakout_rate_at_percentage(df, score_col, breakout_col, .05)
    ten = breakout_rate_at_percentage(df, score_col, breakout_col, .10)
    twenty_five = breakout_rate_at_percentage(df, score_col, breakout_col, .25)
    fifty = breakout_rate_at_percentage(df, score_col, breakout_col, .50)
    
    print(f'Overall Breakout Rate: {fifty['Overall Breakout Rate']:.2%}')
    print(f'Top 5% Breakout Rate: {five['Breakout Rate']:.2%}')
    print(f'Top 10% Breakout Rate: {ten['Breakout Rate']:.2%}')
    print(f'Top 25% Breakout Rate: {twenty_five['Breakout Rate']:.2%}')
    print(f'Top 50% Breakout Rate: {fifty['Breakout Rate']:.2%}')


def create_baseline_breakout_model(analysis_df, training_range, testing_range):
    # Creates breakout model that only utilizes FIP-
    
    train_df = analysis_df[(analysis_df['Season'] >= training_range[0]) & (analysis_df['Season'] <= training_range[1])]
    test_df = analysis_df[(analysis_df['Season'] >= testing_range[0]) & (analysis_df['Season'] <= testing_range[1])]

    fip_model = make_pipeline(
    SplineTransformer(n_knots=5, degree=3),
    LogisticRegression(C=1.0, max_iter=1000)
    )

    fip_model.fit(
        train_df[["FIP-"]],
        train_df["led_to_breakout"]
    )

    test_df["score_fip"] = fip_model.predict_proba(
        test_df[["FIP-"]]
    )[:, 1]
    
    return test_df

def lift_curve(y_true, scores, percentiles=np.linspace(0.01, 1.0, 100)):
    # Creates lift curve to compare model performance
    
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    order = np.argsort(scores)
    y_sorted = y_true[order]

    baseline = y_true.mean()

    lifts = []

    for p in percentiles:
        n = max(1, int(len(y_sorted) * p))
        precision = y_sorted[:n].mean()
        lifts.append(precision / baseline)

    return percentiles, np.array(lifts)