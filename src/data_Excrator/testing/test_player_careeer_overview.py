# test_player_profile.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

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
    
    # Example 1: Extract profile overview for a specific player
    player_id = "49303"  # Anders Albertson's ID
    print(f"\nExtracting profile overview for player {player_id}...")
    
    profile_df = extractor.extract_player_profile(
        player_ids=[player_id]
    )
    
    if profile_df.empty:
        print("No profile data found")
    else:
        print(f"Found profile data for {len(profile_df)} player(s)")
        
        # Display player info
        player_data = profile_df.iloc[0]
        print("\nPlayer Information:")
        print(f"Name: {player_data.get('full_name')}")
        print(f"Country: {player_data.get('country')}")
        
        # Display standings if available
        if 'standings_rank' in player_data and pd.notna(player_data['standings_rank']):
            print(f"FedExCup Rank: {player_data['standings_rank']}")
        
        # Display snapshot info
        snapshot_cols = [col for col in profile_df.columns if col.startswith('snapshot_') and not col.endswith('_desc')]
        print("\nCareer Snapshot:")
        for col in snapshot_cols:
            if pd.notna(player_data[col]) and player_data[col] != "":
                # Get clean title
                title = col.replace('snapshot_', '').replace('_', ' ').title()
                print(f"{title}: {player_data[col]}")
        
        # Display latest season stats
        print("\nLatest Season Stats:")
        latest_cols = [col for col in profile_df.columns if col.startswith('latest_')]
        for col in latest_cols:
            if pd.notna(player_data[col]) and player_data[col] != "":
                # Get clean title
                title = col.replace('latest_', '').replace('_', ' ').title()
                print(f"{title}: {player_data[col]}")
    
    # Example 2: Extract performance data for a specific player
    print(f"\nExtracting performance data for player {player_id}...")
    
    perf_df = extractor.extract_player_performance(
        player_ids=[player_id]
    )
    
    if perf_df.empty:
        print("No performance data found")
    else:
        print(f"Found {len(perf_df)} season-tour combinations")
        
        # Display PGA Tour (R) performance only
        pga_perf = perf_df[perf_df['tour'] == 'R'].sort_values('season', ascending=False)
        print("\nPGA Tour Performance by Season:")
        # Display selected columns
        display_cols = ['season', 'display_season', 'events', 'cuts_made', 'top_10', 'earnings']
        display_cols = [col for col in display_cols if col in pga_perf.columns]
        if not pga_perf.empty:
            print(pga_perf[display_cols])
        
        # Display earnings trend if data available
        if 'earnings' in pga_perf.columns and not pga_perf.empty:
            earnings_data = pga_perf[['season', 'earnings']].dropna()
            if len(earnings_data) > 1:  # Need at least 2 data points
                plt.figure(figsize=(10, 6))
                sns.barplot(x='season', y='earnings', data=earnings_data)
                plt.title(f'PGA Tour Earnings by Season - {player_id}')
                plt.xlabel('Season')
                plt.ylabel('Earnings ($)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                earnings_plot_path = os.path.join(os.path.dirname(__file__), 'player_pga_earnings.png')
                plt.savefig(earnings_plot_path)
                print(f"\nSaved PGA Tour earnings plot to {earnings_plot_path}")
    
    # Example 3: Compare performance across tours
    if not perf_df.empty:
        # Check if we have data for multiple tours
        tours = perf_df['tour'].unique()
        if len(tours) > 1:
            print("\nComparing Performance Across Tours:")
            
            # Group by tour and calculate averages
            tour_comparison = perf_df.groupby('tour').agg({
                'events': 'mean',
                'cuts_made': 'mean',
                'top_10': 'mean',
                'top_25': 'mean',
                'earnings': 'mean'
            }).reset_index()
            
            print(tour_comparison)
            
            # Create a visualization comparing cut percentage by tour
            plt.figure(figsize=(10, 6))
            
            # Calculate cut percentage for each season-tour combination
            perf_df['cut_pct'] = perf_df['cuts_made'] / perf_df['events'] * 100
            
            # Box plot of cut percentage by tour
            sns.boxplot(x='tour', y='cut_pct', data=perf_df)
            plt.title('Cut Making Percentage by Tour')
            plt.xlabel('Tour')
            plt.ylabel('Cut Percentage (%)')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            tour_plot_path = os.path.join(os.path.dirname(__file__), 'player_tour_comparison.png')
            plt.savefig(tour_plot_path)
            print(f"\nSaved tour comparison plot to {tour_plot_path}")
    
    # Example 4: Career progression analysis
    if not perf_df.empty:
        pga_perf = perf_df[perf_df['tour'] == 'R'].sort_values('season')
        
        if len(pga_perf) > 2:  # Need at least 3 seasons for trend
            # Calculate cut percentage and earnings per event
            pga_perf['cut_pct'] = pga_perf['cuts_made'] / pga_perf['events'] * 100
            pga_perf['earnings_per_event'] = pga_perf['earnings'] / pga_perf['events']
            
            # Create a multi-metric plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Cut percentage over time
            if 'cut_pct' in pga_perf.columns:
                sns.lineplot(x='season', y='cut_pct', data=pga_perf, marker='o', ax=ax1)
                ax1.set_title('Cut Making Percentage Over Time')
                ax1.set_ylabel('Cut Percentage (%)')
                ax1.grid(True, alpha=0.3)
            
            # Earnings per event over time
            if 'earnings_per_event' in pga_perf.columns:
                sns.lineplot(x='season', y='earnings_per_event', data=pga_perf, marker='o', color='green', ax=ax2)
                ax2.set_title('Earnings per Event Over Time')
                ax2.set_xlabel('Season')
                ax2.set_ylabel('Earnings per Event ($)')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            progression_plot_path = os.path.join(os.path.dirname(__file__), 'player_career_progression.png')
            plt.savefig(progression_plot_path)
            print(f"\nSaved career progression plot to {progression_plot_path}")
    
    # Save a sample of the data to CSV
    if not profile_df.empty:
        output_path = os.path.join(os.path.dirname(__file__), 'player_profile_sample.csv')
        profile_df.to_csv(output_path, index=False)
        print(f"\nSaved player profile sample to {output_path}")
    
    if not perf_df.empty:
        perf_output_path = os.path.join(os.path.dirname(__file__), 'player_performance_sample.csv')
        perf_df.to_csv(perf_output_path, index=False)
        print(f"\nSaved player performance sample to {perf_output_path}")

if __name__ == "__main__":
    main()