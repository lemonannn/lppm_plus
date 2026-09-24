## Helper Functions

import pandas as pd

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

def calculate_breakouts(FIP_decrease_threshold):
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
                (pitcher_fip <= previous_qualified_year_fip - FIP_decrease_threshold) and 
                (pitcher_fip <= two_qualified_years_fip - FIP_decrease_threshold) and 
                (pitcher_fip <= 90) and (pitcher_year[1] != STARTING_YEAR + 1) 
                ):
                
                breakouts.append(pitcher_year)

    return breakouts