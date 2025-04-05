# test_player_career.py
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
    
    # Example 1: Extract career data for a specific player
    player_id = "49303"  # The player ID from the example
    print(f"\nExtracting career data for player {player_id}...")
    
    career_df = extractor.extract_player_career(
        player_ids=[player_id]
    )
    
    if career_df.empty:
        print("No career data found")
    else:
        print(f"Found career data for {len(career_df)} player(s)")
        
        # Display career summary
        player_data = career_df.iloc[0]
        print("\nCareer Summary:")
        print(f"Events Played: {player_data.get('events')}")
        print(f"PGA Tour Wins: {player_data.get('wins')}")
        print(f"Top 10 Finishes: {player_data.get('top10')}")
        print(f"Top 25 Finishes: {player_data.get('top25')}")
        print(f"Career Earnings: {player_data.get('official_money')}")
        
        # Display career span if available
        if 'first_year' in player_data and 'last_year' in player_data:
            print(f"Career Span: {player_data['first_year']} - {player_data['last_year']} ({player_data.get('career_span')} years)")
    
    # Example 2: Extract yearly data for a specific player
    print(f"\nExtracting yearly data for player {player_id}...")
    
    yearly_df = extractor.extract_player_career_yearly(
        player_ids=[player_id]
    )
    
    if yearly_df.empty:
        print("No yearly data found")
    else:
        print(f"Found data for {len(yearly_df)} seasons")
        
        # Display yearly performance
        print("\nYearly Performance:")
        # Sort by year in descending order
        sorted_df = yearly_df.sort_values('year', ascending=False)
        # Display key columns
        display_cols = ['year', 'display_season', 'events', 'wins', 'top10', 'top25', 'cuts_made', 'official_money']
        display_cols = [col for col in display_cols if col in sorted_df.columns]
        print(sorted_df[display_cols])
        
        # Plot earnings over time if data is available
        if 'year' in sorted_df.columns and 'official_money' in sorted_df.columns:
            # Only include years with earnings data
            plot_data = sorted_df[sorted_df['official_money'].notna()]
            
            if len(plot_data) > 1:  # Only plot if we have multiple data points
                plt.figure(figsize=(10, 6))
                sns.lineplot(x='year', y='official_money', data=plot_data, marker='o')
                plt.title(f'Official Money by Year - Player {player_id}')
                plt.ylabel('Earnings ($)')
                plt.xlabel('Year')
                plt.xticks(plot_data['year'])
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                earnings_plot_path = os.path.join(os.path.dirname(__file__), 'player_earnings.png')
                plt.savefig(earnings_plot_path)
                print(f"\nSaved earnings plot to {earnings_plot_path}")
    
    # Example 3: Extract performance trends over time
    if not yearly_df.empty:
        perf_metrics = ['events', 'cuts_made', 'top10', 'top25']
        available_metrics = [m for m in perf_metrics if m in yearly_df.columns]
        
        if available_metrics and len(yearly_df) > 1:
            # Prepare data for plotting
            plot_df = yearly_df.sort_values('year')
            
            plt.figure(figsize=(12, 7))
            
            for metric in available_metrics:
                if plot_df[metric].notna().any():
                    plt.plot(plot_df['year'], plot_df[metric], marker='o', label=metric)
            
            plt.title(f'Performance Metrics by Year - Player {player_id}')
            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.xticks(plot_df['year'])
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            perf_plot_path = os.path.join(os.path.dirname(__file__), 'player_performance_trends.png')
            plt.savefig(perf_plot_path)
            print(f"\nSaved performance trends plot to {perf_plot_path}")
    
    # Example 4: Compare multiple players
    player_ids = ["49303", "52955", "06527"]  # Multiple players
    print(f"\nComparing career data for {len(player_ids)} players...")
    
    multi_career_df = extractor.extract_player_career(
        player_ids=player_ids
    )
    
    if multi_career_df.empty:
        print("No multi-player career data found")
    else:
        print(f"Found career data for {len(multi_career_df)} players")
        
        # Display career comparison
        print("\nCareer Comparison:")
        comparison_cols = ['player_id', 'events', 'wins', 'top10', 'top25', 'career_span']
        comparison_cols = [col for col in comparison_cols if col in multi_career_df.columns]
        print(multi_career_df[comparison_cols])
        
        # Save the comparison data
        if not multi_career_df.empty:
            output_path = os.path.join(os.path.dirname(__file__), 'player_career_comparison.csv')
            multi_career_df.to_csv(output_path, index=False)
            print(f"\nSaved player career comparison to {output_path}")

if __name__ == "__main__":
    main()