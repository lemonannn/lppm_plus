from fungo import fangraphs
import pandas as pd


def create_dataset():
    pitchers = fangraphs.get_leaders(stats="pit", start_season=2015, end_season=2025, qual=100, ind=1)
    pitchers_df = pd.DataFrame(pitchers)
    df = pitchers_df[['PlayerName', 'FIP-', 'Season', 'piwFA', 'piwFC', 'piwFS', 'piwSI', 'piwCH', 'piwSL', 'piwCU']]
    df.to_csv('arsenal_data.csv')
    
    return df


arsenal_df = create_dataset()