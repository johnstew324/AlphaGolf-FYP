# test_data_extractor.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to allow imports from sibling directories
# This assumes test_data_extractor.py is in alphagolf/alphagolf/src/data_Excator/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now we can import from sibling directories
from database import DatabaseManager
from config import Config
from data_Excator.data_excractor import DataExtractor

def main():
    # Initialize the database manager
    try:
        print(f"Connecting to MongoDB using URI: {Config.MONGODB_URI[:20]}...")
        db_manager = DatabaseManager(uri=Config.MONGODB_URI, database_name="pga_tour_data")
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return

    # Create the data extractor
    extractor = DataExtractor(db_manager)
    
    # List all collections in the database
    collections = extractor.list_collections()
    print(f"Available collections: {collections}")
    
    # Extract player stats for a specific season
    print("Extracting player stats for 2023...")
    player_stats_df = extractor.extract_player_stats(
        seasons=2023,
        stat_categories=["STROKES_GAINED, DRIVING", "STROKES_GAINED, PUTTING"]
    )
    
    # Check if we got any data
    if player_stats_df.empty:
        print("No player stats data found")
        return
        
    # Show the first few rows
    print("\nPlayer Stats Sample:")
    print(player_stats_df.head())
    
    # Get basic stats info
    print("\nDataFrame Info:")
    print(f"Rows: {player_stats_df.shape[0]}, Columns: {player_stats_df.shape[1]}")
    
    # Print column names
    print("\nColumn names:")
    for col in player_stats_df.columns:
        print(f"  - {col}")
    
    # Example: Extract stats for specific players
    print("\nExtracting stats for specific players...")
    top_players = extractor.extract_player_stats(
        seasons=2023,
        player_ids=["52955"],  # Ludvig Åberg's ID
        stat_categories=["STROKES_GAINED, SCORING"]
    )
    
    print("\nTop Player Strokes Gained Total:")
    if not top_players.empty:
        sg_cols = [col for col in top_players.columns if 'sg_total' in col.lower()]
        if sg_cols:
            print(top_players[['name', 'season'] + sg_cols])
        else:
            print("No SG Total columns found")
    
    # Save extracted data to CSV
    if not player_stats_df.empty:
        output_path = os.path.join(os.path.dirname(__file__), 'player_stats_2023.csv')
        player_stats_df.to_csv(output_path, index=False)
        print(f"\nSaved player stats to {output_path}")

if __name__ == "__main__":
    main()