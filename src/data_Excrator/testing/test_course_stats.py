# test_course_stats.py
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
    
    # Example 1: Extract course stats for a specific tournament
    tournament_id = "R2023016"  # Sentry Tournament (2023)
    print(f"\nExtracting course stats for tournament {tournament_id}...")
    
    course_df = extractor.extract_course_stats(
        tournament_ids=tournament_id
    )
    
    if course_df.empty:
        print("No course stats found")
    else:
        print(f"Found stats for {len(course_df)} course(s)")
        
        # Display basic course info
        course_data = course_df.iloc[0]
        print("\nCourse Information:")
        print(f"Name: {course_data.get('course_name')}")
        print(f"Location: {course_data.get('overview_city')}, {course_data.get('overview_state')}")
        print(f"Par: {course_data.get('par')}")
        
        # Display course details from overview
        print("\nCourse Details:")
        overview_cols = [col for col in course_df.columns if col.startswith('overview_') and not col.endswith('_detail')]
        for col in overview_cols:
            if pd.notna(course_data[col]) and col not in ['overview_name', 'overview_city', 'overview_state', 'overview_country']:
                # Get clean title
                title = col.replace('overview_', '').replace('_', ' ').title()
                print(f"{title}: {course_data[col]}")
        
        # Display scoring summary
        print("\nScoring Summary:")
        scoring_cols = ['summary_eagles', 'summary_birdies', 'summary_pars', 
                        'summary_bogeys', 'summary_double_bogeys']
        scoring_cols = [col for col in scoring_cols if col in course_df.columns]
        
        for col in scoring_cols:
            if pd.notna(course_data[col]):
                # Get clean title
                title = col.replace('summary_', '').replace('_', ' ').title()
                print(f"{title}: {course_data[col]}")
    
    # Example 2: Extract hole-level stats
    print(f"\nExtracting hole-level stats for tournament {tournament_id}...")
    
    hole_df = extractor.extract_hole_stats(
        tournament_ids=tournament_id
    )
    
    if hole_df.empty:
        print("No hole stats found")
    else:
        print(f"Found stats for {len(hole_df)} holes")
        
        # Calculate average scoring relative to par by hole
        if 'hole_par' in hole_df.columns and 'hole_scoring_average' in hole_df.columns:
            hole_df['scoring_vs_par'] = hole_df['hole_scoring_average'] - hole_df['hole_par']
            
            # Group by hole number and calculate average across rounds
            hole_summary = hole_df.groupby('hole_number').agg({
                'hole_par': 'first',
                'hole_yards': 'mean',
                'scoring_vs_par': 'mean',
                'hole_birdies': 'sum',
                'hole_bogeys': 'sum'
            }).reset_index()
            
            print("\nHole Difficulty (Average Score Relative to Par):")
            print(hole_summary.sort_values('scoring_vs_par', ascending=False).head(5))
            
            # Create a visualization of hole difficulty
            plt.figure(figsize=(12, 6))
            
            # Create a bar chart of scoring relative to par
            bars = plt.bar(hole_summary['hole_number'], hole_summary['scoring_vs_par'])
            
            # Color bars based on whether they're over or under par
            for i, bar in enumerate(bars):
                if hole_summary.iloc[i]['scoring_vs_par'] > 0:
                    bar.set_color('red')  # Over par
                else:
                    bar.set_color('green')  # Under par
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Hole Difficulty: Average Score Relative to Par')
            plt.xlabel('Hole Number')
            plt.ylabel('Strokes Above/Below Par')
            plt.xticks(hole_summary['hole_number'])
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            difficulty_plot_path = os.path.join(os.path.dirname(__file__), 'hole_difficulty.png')
            plt.savefig(difficulty_plot_path)
            print(f"\nSaved hole difficulty plot to {difficulty_plot_path}")
    
    # Example 3: Extract stats for a specific round and compare holes
    print(f"\nExtracting stats for round 4 of tournament {tournament_id}...")
    
    round4_df = extractor.extract_hole_stats(
        tournament_ids=tournament_id,
        round_numbers=4
    )
    
    if round4_df.empty:
        print("No round 4 stats found")
    else:
        print(f"Found stats for {len(round4_df)} holes in round 4")
        
        # Analyze scoring distribution in round 4
        if 'hole_pars' in round4_df.columns and 'hole_birdies' in round4_df.columns:
            # Calculate birdie percentage and bogey percentage
            round4_df['birdie_pct'] = round4_df['hole_birdies'] / (round4_df['hole_eagles'] + round4_df['hole_birdies'] + round4_df['hole_pars'] + round4_df['hole_bogeys'] + round4_df['hole_double_bogeys']) * 100
            round4_df['bogey_pct'] = (round4_df['hole_bogeys'] + round4_df['hole_double_bogeys']) / (round4_df['hole_eagles'] + round4_df['hole_birdies'] + round4_df['hole_pars'] + round4_df['hole_bogeys'] + round4_df['hole_double_bogeys']) * 100
            
            # Compare par 3s, par 4s, and par 5s
            par_groups = round4_df.groupby('hole_par')
            
            print("\nScoring by Par in Round 4:")
            for par, group in par_groups:
                hole_count = len(group)
                avg_scoring = group['hole_scoring_average'].mean()
                avg_vs_par = avg_scoring - par
                birdie_pct = group['birdie_pct'].mean()
                bogey_pct = group['bogey_pct'].mean()
                
                print(f"Par {par} Holes ({hole_count}):")
                print(f"  Avg Score: {avg_scoring:.2f} ({avg_vs_par:+.2f})")
                print(f"  Birdie %: {birdie_pct:.1f}%")
                print(f"  Bogey+ %: {bogey_pct:.1f}%")
            
            # Create a visualization comparing par types
            if len(par_groups) > 1:
                plt.figure(figsize=(10, 6))
                
                # Calculate summary statistics by par
                par_summary = round4_df.groupby('hole_par').agg({
                    'birdie_pct': 'mean',
                    'bogey_pct': 'mean'
                }).reset_index()
                
                # Create a grouped bar chart
                width = 0.35
                x = par_summary['hole_par']
                
                plt.bar(x - width/2, par_summary['birdie_pct'], width, label='Birdie %', color='green')
                plt.bar(x + width/2, par_summary['bogey_pct'], width, label='Bogey+ %', color='red')
                
                plt.title('Scoring by Par Type in Round 4')
                plt.xlabel('Par')
                plt.ylabel('Percentage')
                plt.xticks(x)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                par_plot_path = os.path.join(os.path.dirname(__file__), 'par_comparison.png')
                plt.savefig(par_plot_path)
                print(f"\nSaved par comparison plot to {par_plot_path}")
    
    # Example 4: Compare multiple courses
    tournament_ids = ["R2023016", "R2023003"]  # Sentry and Masters
    print(f"\nComparing courses across tournaments: {tournament_ids}...")
    
    multi_course_df = extractor.extract_course_stats(
        tournament_ids=tournament_ids
    )
    
    if multi_course_df.empty:
        print("No multi-course data found")
    else:
        print(f"Found data for {len(multi_course_df)} courses")
        
        # Display course comparison
        print("\nCourse Comparison:")
        if 'summary_birdies' in multi_course_df.columns and 'summary_bogeys' in multi_course_df.columns:
            # Calculate birdie to bogey ratio
            multi_course_df['birdie_to_bogey_ratio'] = multi_course_df['summary_birdies'] / (multi_course_df['summary_bogeys'] + multi_course_df['summary_double_bogeys'])
            
            # Display comparison
            comparison_cols = ['course_name', 'par', 'summary_total_yards', 
                              'summary_birdies', 'summary_bogeys', 'birdie_to_bogey_ratio']
            comparison_cols = [col for col in comparison_cols if col in multi_course_df.columns]
            
            print(multi_course_df[comparison_cols])
        
        # Save the course comparison data
        if not multi_course_df.empty:
            output_path = os.path.join(os.path.dirname(__file__), 'course_comparison.csv')
            multi_course_df.to_csv(output_path, index=False)
            print(f"\nSaved course comparison to {output_path}")
    
    # Save a sample of the hole data to CSV
    if not hole_df.empty:
        hole_output_path = os.path.join(os.path.dirname(__file__), 'hole_stats_sample.csv')
        # Select a subset of columns for readability
        display_cols = ['tournament_id', 'course_name', 'round_number', 'hole_number', 
                       'hole_par', 'hole_yards', 'hole_scoring_average', 
                       'hole_birdies', 'hole_bogeys']
        display_cols = [col for col in display_cols if col in hole_df.columns]
        
        hole_df[display_cols].to_csv(hole_output_path, index=False)
        print(f"\nSaved hole stats sample to {hole_output_path}")

if __name__ == "__main__":
    main()