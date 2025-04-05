# test_tournament_history_stats.py
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
    
    # Example 1: Extract tournament history stats for a specific tournament
    tournament_id = "R2025016"  # Use a valid tournament ID from your database
    
    print(f"\nExtracting tournament history stats for tournament {tournament_id}...")
    
    history_df = extractor.extract_tournament_history_stats(
        tournament_id=tournament_id
    )
    
    if history_df.empty:
        print("No tournament history stats found")
    else:
        print(f"Found tournament history stats for {len(history_df)} players")
        
        # Display player count
        player_count = history_df['player_id'].nunique()
        print(f"Number of unique players: {player_count}")
        
        # Show sample data for a player
        sample_player = history_df.iloc[0]
        print("\nSample Player Data:")
        print(f"Player ID: {sample_player['player_id']}")
        print(f"Total Rounds: {sample_player['total_rounds']}")
        
        # Display tournament history results
        print("\nTournament History Results:")
        history_columns = [col for col in sample_player.index if "history" in col and col.endswith("_tournament_name")]
        
        for col in history_columns:
            history_num = col.split("_")[0].replace("history", "")
            tournament_name = sample_player[col]
            
            if pd.notna(tournament_name):
                position = sample_player[f"history{history_num}_position"]
                score = sample_player[f"history{history_num}_score"]
                season = sample_player[f"history{history_num}_season"]
                
                print(f"  Tournament {history_num}: {tournament_name} ({season}) - Position: {position}, Score: {score}")
        
        # Display strokes gained data
        print("\nStrokes Gained Data:")
        sg_columns = [col for col in sample_player.index if col.startswith("sg_") and col.endswith("_value")]
        
        for col in sg_columns:
            category = col.replace("_value", "").upper()
            value = sample_player[col]
            print(f"  {category}: {value if pd.notna(value) else 'N/A'}")
    
    # Example 2: Extract tournament history stats for specific players
    player_ids = ["52955", "28089"]  # Use valid player IDs from your database
    
    print(f"\nExtracting tournament history stats for players {player_ids}...")
    
    player_history_df = extractor.extract_tournament_history_stats(
        player_ids=player_ids
    )
    
    if player_history_df.empty:
        print("No tournament history stats found for specified players")
    else:
        print(f"Found tournament history stats for {player_history_df['player_id'].nunique()} of {len(player_ids)} players")
        
        # Display summary for each player
        for player_id in player_history_df['player_id'].unique():
            player_data = player_history_df[player_history_df['player_id'] == player_id].iloc[0]
            
            print(f"\nPlayer ID: {player_id}")
            print(f"Total Rounds: {player_data['total_rounds']}")
            
            # Display tournament history count
            history_columns = [col for col in player_data.index if "history" in col and col.endswith("_tournament_name")]
            history_count = sum(1 for col in history_columns if pd.notna(player_data[col]))
            
            print(f"Tournament History Records: {history_count}")
            
            # Display strokes gained data if available
            sg_columns = [col for col in player_data.index if col.startswith("sg_") and col.endswith("_value")]
            
            print("Strokes Gained Data:")
            for col in sg_columns:
                category = col.replace("_value", "").upper()
                value = player_data[col]
                print(f"  {category}: {value if pd.notna(value) else 'N/A'}")
    
    # Example 3: Analyze tournament history performance metrics
    if not history_df.empty:
        print("\nAnalyzing tournament history performance metrics...")
        
        # Calculate average position and score
        positions = []
        scores = []
        
        # Collect all positions and scores across all players
        for _, row in history_df.iterrows():
            position_columns = [col for col in row.index if "position" in col]
            score_columns = [col for col in row.index if col.endswith("_score") and "history" in col]
            
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
            
            for score_col in score_columns:
                score = row[score_col]
                if pd.notna(score):
                    scores.append(score)
        
        if positions:
            avg_position = sum(positions) / len(positions)
            median_position = sorted(positions)[len(positions) // 2]
            print(f"Average Position: {avg_position:.2f}")
            print(f"Median Position: {median_position}")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average Score: {avg_score:.2f}")
        
        # Calculate average strokes gained values
        sg_columns = [col for col in history_df.columns if col.startswith("sg_") and col.endswith("_value")]
        
        if sg_columns:
            print("\nAverage Strokes Gained Values:")
            for col in sg_columns:
                category = col.replace("_value", "").upper()
                avg_value = history_df[col].mean()
                print(f"  {category}: {avg_value if pd.notna(avg_value) else 'N/A'}")
        
        # Create a visualization of position distribution
        if positions:
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
            position_plot_path = os.path.join(os.path.dirname(__file__), 'tournament_position_distribution.png')
            plt.savefig(position_plot_path)
            print(f"\nSaved position distribution plot to {position_plot_path}")
    
    # Example 4: Analyze strokes gained values
    sg_df = history_df.copy()
    sg_columns = [col for col in sg_df.columns if col.startswith("sg_") and col.endswith("_value")]
    
    if not sg_df.empty and sg_columns:
        print("\nAnalyzing strokes gained values...")
        
        # Create a correlation matrix of strokes gained values
        sg_correlation = sg_df[sg_columns].corr()
        
        print("Correlation Matrix of Strokes Gained Values:")
        print(sg_correlation)
        
        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(sg_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Strokes Gained Categories')
        plt.tight_layout()
        
        # Save the plot
        correlation_plot_path = os.path.join(os.path.dirname(__file__), 'strokes_gained_correlation.png')
        plt.savefig(correlation_plot_path)
        print(f"Saved correlation matrix plot to {correlation_plot_path}")
        
        # Create a boxplot of strokes gained values
        plt.figure(figsize=(12, 6))
        
        # Prepare the data
        sg_data = sg_df[sg_columns].copy()
        sg_data.columns = [col.replace("_value", "").upper() for col in sg_columns]
        
        # Plot boxplot
        sns.boxplot(data=sg_data)
        plt.title('Distribution of Strokes Gained Categories')
        plt.ylabel('Strokes Gained')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        boxplot_path = os.path.join(os.path.dirname(__file__), 'strokes_gained_boxplot.png')
        plt.savefig(boxplot_path)
        print(f"Saved strokes gained boxplot to {boxplot_path}")

if __name__ == "__main__":
    main() 