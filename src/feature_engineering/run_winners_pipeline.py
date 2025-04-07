"""
run_winners_pipeline.py - Simple script to run the winners feature pipeline
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the winners pipeline
from feature_engineering.winners_pipeline import WinnersFeaturePipeline

def run_pipeline(input_file=None, output_dir=None, use_timestamp=True):
    """
    Run the winners feature pipeline with specified options.
    
    Args:
        input_file: Path to input predictive features file
        output_dir: Directory to save outputs
        use_timestamp: Whether to add a timestamp to output files
    
    Returns:
        Path to the generated winners dataset
    """
    start_time = datetime.now()
    print(f"Starting winners feature pipeline at {start_time}")
    
    # Set default paths if not provided
    if not input_file:
        # Try to find the predictive features file
        candidates = [
            'feature_analysis/predictive_features.csv',
            'predictive_features.csv',
            '../feature_analysis/predictive_features.csv',
            '../predictive_features.csv'
        ]
        
        for path in candidates:
            if os.path.exists(path):
                input_file = path
                break
        
        if not input_file:
            print("Error: Could not find predictive_features.csv")
            return None
    
    if not output_dir:
        output_dir = 'output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp if requested
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"winners_features_{timestamp}.csv"
    else:
        output_file = "winners_features.csv"
    
    # Create and run the pipeline
    try:
        pipeline = WinnersFeaturePipeline(input_file=input_file, output_dir=output_dir)
        winners_data = pipeline.run()
        
        # Get the output path
        output_path = os.path.join(output_dir, output_file)
        
        # Save with the custom filename if it differs from the default
        if output_file != "winners_features.csv":
            winners_data.to_csv(output_path, index=False)
            print(f"Saved winners dataset to {output_path}")
        
        # Calculate and print elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        # Print dataset statistics
        print(f"\nDataset shape: {winners_data.shape[0]} rows, {winners_data.shape[1]} columns")
        
        # Print distribution of target variables
        for target in ['winner', 'top3', 'top10', 'top25']:
            if target in winners_data.columns:
                count = winners_data[target].sum()
                print(f"{target.capitalize()}: {count} examples ({100*count/len(winners_data):.2f}%)")
        
        # Print tournament and player counts
        if 'tournament_id' in winners_data.columns:
            tournament_count = winners_data['tournament_id'].nunique()
            print(f"Tournaments: {tournament_count}")
        
        if 'player_id' in winners_data.columns:
            player_count = winners_data['player_id'].nunique()
            print(f"Players: {player_count}")
        
        return output_path
    
    except Exception as e:
        print(f"Error running winners pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the winners feature pipeline')
    
    parser.add_argument('--input', '-i', 
                        help='Path to input predictive features CSV file')
    
    parser.add_argument('--output-dir', '-o', 
                        help='Directory to save output files')
    
    parser.add_argument('--no-timestamp', '-n', action='store_true',
                        help='Disable timestamp in output filename')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the pipeline
    output_path = run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        use_timestamp=not args.no_timestamp
    )
    
    if output_path:
        print(f"\nWinners dataset created successfully at:\n{output_path}")
    else:
        print("\nError creating winners dataset. See above for details.")
        sys.exit(1)