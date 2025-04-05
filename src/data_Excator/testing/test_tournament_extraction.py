# test_tournament_history.py
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
    
    # Example 1: Extract tournament history for a specific tournament across multiple years
    tournament_id = "R2025016"  # Sentry Tournament of Champions
    print(f"Extracting tournament history for {tournament_id}...")
    
    tournament_df = extractor.extract_tournament_history(
        tournament_ids=tournament_id
    )
    
    if tournament_df.empty:
        print("No tournament history found")
    else:
        print(f"Found {len(tournament_df)} years of tournament history")
        print("\nRecent tournament winners:")
        # Sort by year and show most recent 5 years
        recent_winners = tournament_df.sort_values('year', ascending=False).head(5)
        print(recent_winners[['year', 'winner_name', 'winning_score_to_par']])
    
    # Example 2: Extract a player's performance at a specific tournament
    player_id = "06527"  # Steve Stricker
    print(f"\nExtracting tournament history for player {player_id} at tournament {tournament_id}...")
    
    player_tournament_df = extractor.extract_tournament_history(
        tournament_ids=tournament_id,
        player_ids=[player_id]
    )
    
    if player_tournament_df.empty:
        print("No tournament history found for this player")
    else:
        print(f"Found {len(player_tournament_df)} tournament appearances")
        print("\nPlayer tournament performances:")
        # Show performance sorted by year
        performances = player_tournament_df.sort_values('year', ascending=False)
        print(performances[['year', 'player_name', 'position', 'total_score', 'score_to_par']])
    
    # Example 3: Find a player's tournament performance statistics
    print(f"\nCalculating tournament performance stats for player {player_id}...")
    
    performance_stats = extractor.extract_tournament_performance_stats(
        player_id=player_id, 
        min_years=2  # Require at least 2 appearances at a tournament
    )
    
    if performance_stats.empty:
        print("No tournament performance stats found")
    else:
        print(f"Found stats for {len(performance_stats)} tournaments")
        print("\nTop tournaments by average finish:")
        # Sort by average finish (lower is better)
        top_tournaments = performance_stats.sort_values('avg_finish').head(5)
        print(top_tournaments[['tournament_id', 'appearances', 'avg_finish', 'best_finish', 'top_10_finishes']])
    
    # Example 4: Extract multiple players at multiple tournaments
    player_ids = ["06527", "20850", "08793"]  # Stricker, Chopra, and another player
    tournament_ids = ["R2025016", "R2025003"]  # Sentry and The Masters
    print(f"\nExtracting tournament history for {len(player_ids)} players at {len(tournament_ids)} tournaments...")
    
    multi_df = extractor.extract_tournament_history(
        tournament_ids=tournament_ids,
        player_ids=player_ids
    )
    
    if multi_df.empty:
        print("No multi-player tournament history found")
    else:
        print(f"Found {len(multi_df)} player-tournament combinations")
        print("\nPlayer performances:")
        print(multi_df[['year', 'tournament_id', 'player_name', 'position', 'total_score']])
    
    # Save a sample of the data to CSV
    if not tournament_df.empty:
        output_path = os.path.join(os.path.dirname(__file__), 'tournament_history_sample.csv')
        tournament_df.to_csv(output_path, index=False)
        print(f"\nSaved tournament history sample to {output_path}")

if __name__ == "__main__":
    main()