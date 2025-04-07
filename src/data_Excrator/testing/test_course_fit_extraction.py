# test_course_fit_extraction.py
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
    
    # Example 1: Extract course fit data for a specific tournament
    tournament_id = "R2025016"  # Use a valid tournament ID from your database
    
    print(f"\nExtracting course fit data for tournament {tournament_id}...")
    
    course_fit_df = extractor.extract_course_fit(
        tournament_id=tournament_id
    )
    
    if course_fit_df.empty:
        print("No course fit data found")
    else:
        print(f"Found course fit data for {len(course_fit_df)} players")
        
        # Display player count
        player_count = course_fit_df['player_id'].nunique()
        print(f"Number of unique players: {player_count}")
        
        # Show sample data for a player
        sample_player = course_fit_df.iloc[0]
        print("\nSample Player Data:")
        print(f"Player ID: {sample_player['player_id']}")
        print(f"Total Rounds: {sample_player['total_rounds']}")
        print(f"Score: {sample_player['score']}")
        
        # Display course fit stats
        print("\nCourse Fit Stats:")
        value_columns = [col for col in sample_player.index if col.endswith("_value")]
        rank_columns = [col for col in sample_player.index if col.endswith("_rank")]
        
        for col in value_columns:
            stat_name = col.replace("_value", "").replace("_", " ").title()
            value = sample_player[col]
            rank = sample_player[col.replace("_value", "_rank")] if col.replace("_value", "_rank") in sample_player else None
            color = sample_player[col.replace("_value", "_color")] if col.replace("_value", "_color") in sample_player else None
            
            rank_text = f" (Rank: {int(rank)})" if pd.notna(rank) else ""
            color_text = f" [{color}]" if pd.notna(color) else ""
            
            print(f"  {stat_name}: {value:.3f}{rank_text}{color_text}")
    
    # Example 2: Extract course fit data for specific players
    player_ids = ["59141", "28089"]  # Use valid player IDs from your database
    
    print(f"\nExtracting course fit data for players {player_ids}...")
    
    player_fit_df = extractor.extract_course_fit(
        player_ids=player_ids
    )
    
    if player_fit_df.empty:
        print("No course fit data found for specified players")
    else:
        print(f"Found course fit data for {player_fit_df['player_id'].nunique()} of {len(player_ids)} players")
        
        # Display summary for each player
        for player_id in player_fit_df['player_id'].unique():
            player_data = player_fit_df[player_fit_df['player_id'] == player_id].iloc[0]
            
            print(f"\nPlayer ID: {player_id}")
            print(f"Total Rounds: {player_data['total_rounds']}")
            print(f"Score: {player_data['score']}")
            
            # Display top 3 stats (based on rank)
            value_columns = [col for col in player_data.index if col.endswith("_value")]
            rank_columns = [col for col in player_data.index if col.endswith("_rank")]
            
            # Create list of (stat_name, value, rank) tuples for sorting
            stats = []
            for col in value_columns:
                stat_name = col.replace("_value", "").replace("_", " ").title()
                value = player_data[col]
                rank_col = col.replace("_value", "_rank")
                
                if rank_col in player_data:
                    rank = player_data[rank_col]
                    stats.append((stat_name, value, rank))
            
            # Sort by rank (ascending) and get top 3
            stats.sort(key=lambda x: x[2] if pd.notna(x[2]) else float('inf'))
            top_stats = stats[:3]
            
            print("\nTop 3 Course Fit Stats:")
            for stat_name, value, rank in top_stats:
                print(f"  {stat_name}: {value:.3f} (Rank: {int(rank)})")
    
    # Example 3: Analyze course fit data across all players
    if not course_fit_df.empty:
        print("\nAnalyzing course fit data across all players...")
        
        # Identify all stat value columns
        value_columns = [col for col in course_fit_df.columns if col.endswith("_value")]
        
        # Calculate average values for each stat
        stat_averages = {}
        for col in value_columns:
            stat_name = col.replace("_value", "").replace("_", " ").title()
            avg_value = course_fit_df[col].mean()
            stat_averages[stat_name] = avg_value
        
        print("Average Values by Stat Category:")
        for stat_name, avg in stat_averages.items():
            print(f"  {stat_name}: {avg:.3f}")
        
        # Create a visualization of stat distributions
        plt.figure(figsize=(14, 8))
        
        # Create a boxplot for each stat category
        stat_data = course_fit_df[value_columns]
        stat_data.columns = [col.replace("_value", "").replace("_", " ").title() for col in value_columns]
        
        # Plot boxplot
        sns.boxplot(data=stat_data)
        plt.title('Distribution of Course Fit Stats Across Players')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Stat Value')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        stat_plot_path = os.path.join(os.path.dirname(__file__), 'course_fit_distribution.png')
        plt.savefig(stat_plot_path)
        print(f"\nSaved course fit distribution plot to {stat_plot_path}")
    
    # Example 4: Analyze correlation between stats and score
    if not course_fit_df.empty and 'score' in course_fit_df.columns:
        print("\nAnalyzing correlation between stats and score...")
        
        # Calculate correlation coefficients
        value_columns = [col for col in course_fit_df.columns if col.endswith("_value")]
        correlations = []
        
        for col in value_columns:
            stat_name = col.replace("_value", "").replace("_", " ").title()
            corr = course_fit_df[['score', col]].corr().iloc[0, 1]
            if pd.notna(corr):
                correlations.append((stat_name, corr))
        
        # Sort by absolute correlation (descending)
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Correlation between Stats and Score:")
        for stat_name, corr in correlations:
            direction = "positive" if corr > 0 else "negative"
            print(f"  {stat_name}: {corr:.3f} ({direction})")
        
        # Create a bar chart of correlations
        plt.figure(figsize=(12, 8))
        
        stat_names = [x[0] for x in correlations]
        corr_values = [x[1] for x in correlations]
        
        bars = plt.bar(stat_names, corr_values)
        
        # Color bars based on correlation direction
        for i, bar in enumerate(bars):
            if corr_values[i] > 0:
                bar.set_color('orange')  # Positive correlation
            else:
                bar.set_color('blue')  # Negative correlation
        
        plt.title('Correlation between Course Fit Stats and Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Correlation Coefficient')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        corr_plot_path = os.path.join(os.path.dirname(__file__), 'course_fit_correlation.png')
        plt.savefig(corr_plot_path)
        print(f"Saved correlation plot to {corr_plot_path}")

if __name__ == "__main__":
    main() 