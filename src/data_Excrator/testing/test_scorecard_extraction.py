
# test_scorecards.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # data_Excator folder
src_dir = os.path.dirname(parent_dir)  # src folder
sys.path.append(src_dir)

# Import required modules
from database import DatabaseManager
from config import Config
from data_Excator.data_excractor import DataExtractor

def main():
    # Initialize the database manager
    try:
        print(f"Connecting to MongoDB...")
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return

    # Create the data extractor
    extractor = DataExtractor(db_manager)
    
    # Example 1: Extract round-by-round scorecard data for a specific tournament
    tournament_id = "R2025016"  # The Sentry tournament
    print(f"\nExtracting round-by-round scorecard data for tournament {tournament_id}...")
    
    scorecards_df = extractor.extract_player_scorecards(
        tournament_ids=tournament_id
    )
    
    if scorecards_df.empty:
        print("No scorecard data found")
    else:
        print(f"Found {len(scorecards_df)} player-round records")
        print("\nSample of round scores:")
        # Show a sample of round scores
        sample_rounds = scorecards_df.head(10)
        print(sample_rounds[['player_name', 'round_number', 'round_total', 'score_to_par']])
        
        # Calculate average scores by round
        print("\nAverage scores by round:")
        round_avg = scorecards_df.groupby('round_number')[['round_total', 'score_to_par']].mean()
        print(round_avg)
    
    # Example 2: Extract hole-by-hole data for a specific player
    player_id = "33948"  # Byeong Hun An
    print(f"\nExtracting hole-by-hole scorecard data for player {player_id} at tournament {tournament_id}...")
    
    player_holes_df = extractor.extract_player_hole_scores(
        tournament_ids=tournament_id,
        player_ids=player_id
    )
    
    if player_holes_df.empty:
        print("No hole-by-hole data found for this player")
    else:
        print(f"Found {len(player_holes_df)} hole records")
        print("\nFirst round holes sample:")
        # Show first 5 holes of round 1
        first_round_sample = player_holes_df[player_holes_df['round_number'] == 1].head(5)
        print(first_round_sample[['round_number', 'hole_number', 'hole_par', 'hole_score', 'hole_status', 'running_score']])
        
        # Calculate performance by par type
        print("\nPerformance by par type:")
        par_type_stats = player_holes_df.groupby('hole_par')[['hole_score']].agg(['mean', 'count'])
        print(par_type_stats)
    
    # Example 3: Calculate round statistics for a player
    print(f"\nCalculating round statistics for player {player_id}...")
    
    if not player_holes_df.empty:
        round_stats_df = extractor.calculate_player_round_stats(player_holes_df)
        
        if round_stats_df.empty:
            print("No round statistics calculated")
        else:
            print(f"Generated statistics for {len(round_stats_df)} rounds")
            print("\nDetailed round statistics:")
            print(round_stats_df[['round_number', 'round_total', 'score_to_par', 
                                'eagles', 'birdies', 'pars', 'bogeys', 'double_bogeys']])
            
            # Compare performance on par 3s, 4s, and 5s
            print("\nPar type performance:")
            par_cols = [col for col in round_stats_df.columns if col.startswith('par') and col.endswith('_to_par')]
            if par_cols:
                print(round_stats_df[['round_number'] + par_cols])
    
    # Example 4: Analyze a specific round of a tournament
    round_number = 4  # Final round
    print(f"\nAnalyzing round {round_number} of tournament {tournament_id}...")
    
    round_holes_df = extractor.extract_player_hole_scores(
        tournament_ids=tournament_id,
        round_numbers=round_number
    )
    
    if round_holes_df.empty:
        print(f"No hole data found for round {round_number}")
    else:
        print(f"Found {len(round_holes_df)} hole records for round {round_number}")
        
        # Calculate tournament statistics for the round
        tournament_stats_df = extractor.calculate_tournament_stats(
            tournament_id=tournament_id,
            hole_data=round_holes_df
        )
        
        if not tournament_stats_df.empty:
            print("\nHole difficulty ranking (based on average score to par):")
            difficulty = tournament_stats_df.sort_values('score_to_par', ascending=False).head(5)
            print(difficulty[['hole_number', 'hole_par', 'hole_yardage', 'average_score', 'score_to_par']])
            
            # Show distribution of scores on the most difficult hole
            if not difficulty.empty:
                hardest_hole = difficulty.iloc[0]['hole_number']
                print(f"\nScore distribution on hole #{hardest_hole}:")
                score_cols = [col for col in tournament_stats_df.columns if col.endswith('_count')]
                if score_cols:
                    hardest_hole_stats = tournament_stats_df[tournament_stats_df['hole_number'] == hardest_hole]
                    print(hardest_hole_stats[score_cols])
    
    # Example 5: Compare two players' performances in a round
    player_id2 = "30925"  # Another player (example ID)
    print(f"\nComparing performances of players {player_id} and {player_id2} in round {round_number}...")
    
    compare_df = extractor.extract_player_hole_scores(
        tournament_ids=tournament_id,
        player_ids=[player_id, player_id2],
        round_numbers=round_number
    )
    
    if compare_df.empty:
        print("No comparison data found")
    else:
        # Pivot to show both players side by side
        if 'hole_score' in compare_df.columns and 'player_name' in compare_df.columns:
            pivot_df = compare_df.pivot_table(
                index=['hole_number', 'hole_par', 'hole_yardage'],
                columns='player_name',
                values='hole_score'
            )
            print("\nHole-by-hole comparison:")
            print(pivot_df.head(10))
    
    # Save a sample of the data to CSV
    if not player_holes_df.empty:
        output_path = os.path.join(os.path.dirname(__file__), 'scorecard_sample.csv')
        player_holes_df.to_csv(output_path, index=False)
        print(f"\nSaved player hole-by-hole data sample to {output_path}")

if __name__ == "__main__":
    main()