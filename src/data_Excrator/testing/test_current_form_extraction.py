# test_current_form_extraction.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the required directories to sys.path
# Go up two levels from the current file location (from testing folder to src folder's parent)
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
    
    # Example 1: Extract current form for a specific tournament
    tournament_id = "R2024016"  # Use a valid tournament ID from your database
    
    print(f"\nExtracting current form data for tournament {tournament_id}...")
    
    current_form_df = extractor.extract_current_form(
        tournament_id=tournament_id
    )
    
    if current_form_df.empty:
        print("No current form data found")
    else:
        print(f"Found current form data for {len(current_form_df)} players")
        
        # Display player count
        player_count = current_form_df['player_id'].nunique()
        print(f"Number of unique players: {player_count}")
        
        # Show sample data for a player
        sample_player = current_form_df.iloc[0]
        print("\nSample Player Data:")
        print(f"Player ID: {sample_player['player_id']}")
        print(f"Total Rounds: {sample_player['total_rounds']}")
        
        # Display last 5 tournament results
        print("\nLast 5 Tournament Results:")
        for i in range(1, 6):
            tournament_name = sample_player.get(f"last{i}_tournament_name")
            if pd.notna(tournament_name):
                position = sample_player.get(f"last{i}_position")
                score = sample_player.get(f"last{i}_score")
                print(f"  Tournament {i}: {tournament_name} - Position: {position}, Score: {score}")
        
        # Display strokes gained data
        print("\nStrokes Gained Data:")
        sg_columns = [col for col in sample_player.index if col.startswith("sg_") and col.endswith("_value")]
        for col in sg_columns:
            category = col.replace("_value", "").upper()
            value = sample_player[col]
            print(f"  {category}: {value:.3f}")
    
    # Example 2: Extract current form for specific players
    player_ids = ["52955", "28089"]  # Use valid player IDs from your database
    
    print(f"\nExtracting current form data for players {player_ids}...")
    
    player_form_df = extractor.extract_current_form(
        player_ids=player_ids
    )
    
    if player_form_df.empty:
        print("No current form data found for specified players")
    else:
        print(f"Found current form data for {player_form_df['player_id'].nunique()} of {len(player_ids)} players")
        
        # Display summary for each player
        for player_id in player_form_df['player_id'].unique():
            player_data = player_form_df[player_form_df['player_id'] == player_id].iloc[0]
            
            print(f"\nPlayer ID: {player_id}")
            print(f"Total Rounds: {player_data['total_rounds']}")
            
            # Calculate average score from last 5 tournaments
            score_columns = [col for col in player_data.index if col.endswith("_score")]
            scores = [player_data[col] for col in score_columns if pd.notna(player_data[col])]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"Average Score (Last 5 Tournaments): {avg_score:.2f}")
            
            # Display strokes gained data
            sg_columns = [col for col in player_data.index if col.startswith("sg_") and col.endswith("_value")]
            for col in sg_columns:
                category = col.replace("_value", "").upper()
                value = player_data[col]
                print(f"  {category}: {value:.3f}")
    
    # Example 3: Analyze strokes gained data across all players
    if not current_form_df.empty:
        print("\nAnalyzing strokes gained data across all players...")
        
        # Identify all strokes gained columns
        sg_value_columns = [col for col in current_form_df.columns if col.startswith("sg_") and col.endswith("_value")]
        
        # Calculate average strokes gained for each category
        sg_averages = {}
        for col in sg_value_columns:
            category = col.replace("_value", "").upper()
            avg_value = current_form_df[col].mean()
            sg_averages[category] = avg_value
        
        print("Average Strokes Gained by Category:")
        for category, avg in sg_averages.items():
            print(f"  {category}: {avg:.3f}")
        
        # Create a visualization of strokes gained distributions
        plt.figure(figsize=(12, 8))
        
        # Create a boxplot for each strokes gained category
        sg_data = current_form_df[sg_value_columns]
        sg_data.columns = [col.replace("_value", "").upper() for col in sg_value_columns]
        
        # Plot boxplot
        sns.boxplot(data=sg_data)
        plt.title('Distribution of Strokes Gained Categories Across Players')
        plt.ylabel('Strokes Gained')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        sg_plot_path = os.path.join(os.path.dirname(__file__), 'strokes_gained_distribution.png')
        plt.savefig(sg_plot_path)
        print(f"\nSaved strokes gained distribution plot to {sg_plot_path}")
        
        # Create a histogram for total strokes gained
        plt.figure(figsize=(10, 6))
        
        total_sg_col = next((col for col in sg_value_columns if "tot" in col), None)
        if total_sg_col:
            sns.histplot(current_form_df[total_sg_col], kde=True)
            plt.title('Distribution of Total Strokes Gained')
            plt.xlabel('Strokes Gained Total')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            total_sg_plot_path = os.path.join(os.path.dirname(__file__), 'total_strokes_gained_distribution.png')
            plt.savefig(total_sg_plot_path)
            print(f"Saved total strokes gained distribution plot to {total_sg_plot_path}")
    
    # Example 4: Analyze tournament performance patterns
    if not current_form_df.empty:
        print("\nAnalyzing tournament performance patterns...")
        
        # Collect all positions and convert to numeric where possible
        positions = []
        position_columns = [col for col in current_form_df.columns if col.endswith("_position")]
        
        for _, row in current_form_df.iterrows():
            for pos_col in position_columns:
                pos = row[pos_col]
                if pd.notna(pos):
                    # Try to convert positions like "T10" to numeric values
                    if isinstance(pos, str):
                        if pos.startswith("T"):
                            try:
                                positions.append(int(pos[1:]))
                            except ValueError:
                                pass
                        elif pos.isdigit():
                            positions.append(int(pos))
        
        if positions:
            # Calculate distribution statistics
            avg_position = sum(positions) / len(positions)
            median_position = sorted(positions)[len(positions) // 2]
            
            print(f"Average Position: {avg_position:.2f}")
            print(f"Median Position: {median_position}")
            
            # Create a histogram of positions
            plt.figure(figsize=(10, 6))
            sns.histplot(positions, bins=20, kde=True)
            plt.title('Distribution of Tournament Positions')
            plt.xlabel('Position')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add vertical lines for average and median
            plt.axvline(avg_position, color='red', linestyle='--', label=f'Average: {avg_position:.2f}')
            plt.axvline(median_position, color='green', linestyle='--', label=f'Median: {median_position}')
            plt.legend()
            
            # Save the plot
            position_plot_path = os.path.join(os.path.dirname(__file__), 'position_distribution.png')
            plt.savefig(position_plot_path)
            print(f"Saved position distribution plot to {position_plot_path}")

if __name__ == "__main__":
    main() 