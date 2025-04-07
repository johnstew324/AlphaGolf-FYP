# test_tournament_weather.py
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
    
    # Example 1: Extract tournament weather data for a specific tournament
    tournament_id = "R2022003"  # WM Phoenix Open
    print(f"\nExtracting weather data for tournament {tournament_id}...")
    
    weather_df = extractor.extract_tournament_weather(
        tournament_ids=tournament_id
    )
    
    if weather_df.empty:
        print("No weather data found")
    else:
        print(f"Found weather data for {len(weather_df)} tournament(s)")
        
        # Display tournament info
        print("\nTournament Information:")
        print(f"Tournament: {weather_df.iloc[0]['tournament_name']}")
        print(f"Course: {weather_df.iloc[0]['course_name']}")
        print(f"Location: {weather_df.iloc[0]['location']}")
        print(f"Year: {weather_df.iloc[0]['year']}")
        
        # Display average weather conditions
        print("\nAverage Weather Conditions:")
        avg_cols = [col for col in weather_df.columns if col.startswith('avg_')]
        for col in avg_cols:
            metric_name = col.replace('avg_', '')
            print(f"{metric_name.capitalize()}: {weather_df.iloc[0][col]:.1f}")
        
        # If total precipitation exists, show it
        if 'total_precip' in weather_df.columns:
            print(f"Total Precipitation: {weather_df.iloc[0]['total_precip']:.2f}")
    
    # Example 2: Extract round-by-round weather data
    print(f"\nExtracting round-by-round weather data for tournament {tournament_id}...")
    
    rounds_df = extractor.extract_tournament_weather_by_round(
        tournament_ids=tournament_id
    )
    
    if rounds_df.empty:
        print("No round-by-round weather data found")
    else:
        print(f"Found weather data for {len(rounds_df)} rounds")
        
        # Display round-by-round temperatures
        print("\nRound-by-Round Temperatures:")
        print(rounds_df[['round_number', 'datetime', 'temp', 'tempmin', 'tempmax', 'conditions']])
        
        # Plot temperature ranges by round
        if len(rounds_df) > 1:
            plt.figure(figsize=(10, 6))
            
            # Create x-axis labels with round numbers
            x_labels = [f"Round {round_num}" for round_num in rounds_df['round_number']]
            x = range(len(x_labels))
            
            # Plot temperature range
            plt.bar(x, rounds_df['tempmax'] - rounds_df['tempmin'], bottom=rounds_df['tempmin'], 
                   color='lightblue', label='Temp Range')
            
            # Plot average temperature line
            plt.plot(x, rounds_df['temp'], 'ro-', label='Avg Temp')
            
            plt.xticks(x, x_labels)
            plt.title(f"Temperature by Round - {tournament_id}")
            plt.ylabel('Temperature (°F)')
            plt.legend()
            
            # Save the plot
            plot_path = os.path.join(os.path.dirname(__file__), 'tournament_temps.png')
            plt.savefig(plot_path)
            print(f"\nSaved temperature plot to {plot_path}")
    
    # Example 3: Extract weather data for multiple tournaments and years
    tournament_ids = ["R2022003", "R2022007", "R2022016"]  # Multiple tournaments
    print(f"\nExtracting weather data for {len(tournament_ids)} tournaments...")
    
    multi_weather_df = extractor.extract_tournament_weather(
        tournament_ids=tournament_ids
    )
    
    if multi_weather_df.empty:
        print("No multi-tournament weather data found")
    else:
        print(f"Found weather data for {len(multi_weather_df)} tournaments")
        
        # Compare average temperatures across tournaments
        print("\nTournament Temperature Comparison:")
        print(multi_weather_df[['tournament_name', 'year', 'avg_temp', 'avg_windspeed', 'avg_humidity']])
        
        # Save the multi-tournament data
        if not multi_weather_df.empty:
            output_path = os.path.join(os.path.dirname(__file__), 'tournament_weather_comparison.csv')
            multi_weather_df.to_csv(output_path, index=False)
            print(f"\nSaved multi-tournament weather data to {output_path}")
    
    # Example 4: Compare wind conditions across tournaments
    if not multi_weather_df.empty and len(multi_weather_df) > 1:
        plt.figure(figsize=(10, 6))
        
        # Create a bar chart of average wind speeds
        sns.barplot(x='tournament_name', y='avg_windspeed', data=multi_weather_df)
        plt.title('Average Wind Speed by Tournament')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        wind_plot_path = os.path.join(os.path.dirname(__file__), 'tournament_wind_comparison.png')
        plt.savefig(wind_plot_path)
        print(f"\nSaved wind comparison plot to {wind_plot_path}")

if __name__ == "__main__":
    main()