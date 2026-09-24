from fungo import fangraphs
import pandas as pd
import argparse


def create_dataset(start_year, end_year, IP):
    print(f"Getting data from {start_year} to {end_year}")
    
    pitcher_dfs = []
    
    for chunk_start in range(start_year, end_year + 1, 10):
        chunk_end = min(chunk_start + 9, end_year)

        print(f"Loading chunk from {chunk_start} to {chunk_end}")
        
        pitchers = fangraphs.get_leaders(stats="pit", start_season=chunk_start, end_season=chunk_end, qual=IP, ind=1)
        pitchers_df = pd.DataFrame(pitchers)
        
        pitcher_dfs.append(pitchers_df)
    
    final_df = pd.concat(pitcher_dfs, ignore_index=True)

    
    final_df.to_csv('arsenal_data.csv')
    
    return final_df

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("start_year", nargs='?', default=2015, type=int)
    parser.add_argument("end_year", nargs='?', default=2025, type=int)
    parser.add_argument("innings_pitched", nargs='?', default=100, type=float)

    args = parser.parse_args()
    
    create_dataset(args.start_year, args.end_year, args.innings_pitched)
    
if __name__ == "__main__":
    main()