## Helper Functions

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from pygam import LinearGAM, s, terms
from sklearn.linear_model import LinearRegression
import warnings


STARTING_YEAR = pd.read_csv('arsenal_data.csv')['Season'].min()

def create_pitcher_dict():
    ## Used in calculate_breakouts()
    df = pd.read_csv('arsenal_data.csv')
    pitcher_dict = {}
    simple_df = df[['PlayerName', 'Season', 'FIP-']].copy()
    simple_df = simple_df.sort_values(['PlayerName', 'Season'])
    for row in simple_df.itertuples(index=False):

        pitcher = row.PlayerName
        
        if pitcher not in pitcher_dict:
            pitcher_dict[pitcher] = []
            
        pitcher_dict[pitcher].append(list(row))
        
    return pitcher_dict

def calculate_breakouts(FIP_percent_decrease_threshold, FIP_overall_threshold):
    ## Returns list of breakouts by [Pitcher, Year, FIP-]
    
    pitcher_dict = create_pitcher_dict()
    
    breakouts = []

    for pitcher_career in pitcher_dict.values():
        
        previous_qualified_year_fip = 0
        two_qualified_years_fip = 200
        
        for i in range(1, len(pitcher_career)):
            
            pitcher_year = pitcher_career[i]
            
            pitcher_fip = pitcher_year[2]
            previous_qualified_year_fip = pitcher_career[i-1][2]
            if (i - 2 > 0):
                two_qualified_years_fip = pitcher_career[i-2][2]
                
            if ( 
                (FIP_percent_decrease_threshold <= (1 - pitcher_fip / previous_qualified_year_fip)) and 
                # (pitcher_fip <= two_qualified_years_fip - FIP_decrease_threshold) and 
                (pitcher_fip <= FIP_overall_threshold) and (pitcher_year[1] != STARTING_YEAR + 1) 
                ):
                
                breakouts.append(pitcher_year)

    return breakouts

def get_analysis_df(FIP_decrease_threshold, FIP_overall_threshold):
    df = pd.read_csv('arsenal_data.csv')
    
    # basic info, result stats
    # pitch velocity info
    # pitch usage info
    # pitch movement info (horizontal)
    # pitch movement info (vertical)
    filtered_df = df[['PlayerName', 'Season', 'FIP-', 'ERA-', 
                    'pivFA', 'pivFC', 'pivFS', 'pivSI', 'pivCH', 'pivSL', 'pivCU', 'pivCS', 'pivKN',
                    'piFA%', 'piFC%', 'piFS%', 'piSI%', 'piCH%', 'piSL%', 'piCU%', 'piCS%', 'piKN%',
                    'piFA-X', 'piFC-X', 'piFS-X', 'piSI-X', 'piCH-X', 'piSL-X', 'piCU-X', 'piCS-X', 'piKN-X',
                    'piFA-Z', 'piFC-Z', 'piFS-Z', 'piSI-Z', 'piCH-Z', 'piSL-Z', 'piCU-Z', 'piCS-Z', 'piKN-Z',
                    'pb_command', 'pb_c_CH', 'pb_c_CU', 'pb_c_FF', 'pb_c_SI', 'pb_c_SL', 'pb_c_KC', 'pb_c_FC', 'pb_c_FS']].copy()

    breakouts = calculate_breakouts(FIP_decrease_threshold, FIP_overall_threshold)

    filtered_df = filtered_df.sort_values(['PlayerName', 'Season'])
    
    next = filtered_df.groupby("PlayerName").shift(-1)
    next_next = filtered_df.groupby("PlayerName").shift(-2)
    next_next_next = filtered_df.groupby("PlayerName").shift(-3)
    
    filtered_df['next_Season'] = next['Season']
    filtered_df['next_next_Season'] = next_next['Season']
    filtered_df['next_next_next_Season'] = next_next_next['Season']
    
    filtered_df.loc[
            filtered_df['next_next_Season'] <= filtered_df['Season'],
            'next_next_Season'
    ] = 0
    filtered_df.loc[
        filtered_df['next_next_next_Season'] <= filtered_df['Season'],
        'next_next_next_Season'
    ] = 0
    
    breakout_years = {
            (x[0], x[1])
            for x in breakouts
    }
    
    filtered_df["led_to_breakout"] = [
            ((PlayerName, Season) in breakout_years)
            for PlayerName, Season in zip(filtered_df["PlayerName"], filtered_df["next_Season"]) 
            or zip(filtered_df["PlayerName"], filtered_df["next_next_Season"])
            or zip(filtered_df["PlayerName"], filtered_df["next_next_next_Season"])
    ]
    
    filtered_df['next_FIP-'] = next['FIP-']
    
    analysis_df = filtered_df[filtered_df['Season'] >= 2007].copy()

    analysis_df = analysis_df.dropna(subset=["FIP-"])
    analysis_df = analysis_df.dropna(subset=["next_FIP-"])
    
    return analysis_df

def get_train_test_split(analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False):
    train_mask = analysis_df["Season"] < 2020
    test_mask = analysis_df["Season"] > 2020

    pitch_types = ["FA", "FC", "SI", "CH", "SL", "CU"]
    
    feature_cols = []
    
    if velo:
        feature_cols.extend([f"piv{p}" for p in pitch_types])
    if fip:
        feature_cols.extend(["FIP-"])
    if movement:
        feature_cols.extend([f"pi{p}-X" for p in pitch_types] + [f"pi{p}-Z" for p in pitch_types])
    if usage: 
        feature_cols.extend( [f"pi{p}%" for p in pitch_types])
    if movement_residuals:
        feature_cols.extend([f"{p}_X_residual" for p in pitch_types] + [f"{p}_Z_residual" for p in pitch_types])

    # Features and target
    X = analysis_df[feature_cols].fillna(0)
    y = analysis_df["next_FIP-"]


    # Same time-based split you've been using
    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    return X, X_train, X_test, y_train, y_test

def eval_model(y_test, pred):
    # Evaluation
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

def create_random_forest(analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False):
    
    X, X_train, X_test, y_train, y_test = get_train_test_split(analysis_df, velo, fip, movement, usage, movement_residuals)
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Predictions
    pred = rf.predict(X_test)
    
    eval_model(y_test, pred)
    
    pred = rf.predict(X)

    return rf, pred

def create_gam(analysis_df, velo=True, fip=True, movement=True, usage=True, movement_residuals=False):
    
    X, X_train, X_test, y_train, y_test = get_train_test_split(analysis_df, velo, fip, movement, usage, movement_residuals)

    gam_terms = terms.s(0)

    for i in range(1, X_train.shape[1]):
        gam_terms += terms.s(i)

    gam = LinearGAM(gam_terms)
    gam.fit(X_train, y_train)

    pred = gam.predict(X_test)
    
    eval_model(y_test, pred)
    
    pred = gam.predict(X)
    
    return gam, pred


def predict_residuals(analysis_df):
    
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