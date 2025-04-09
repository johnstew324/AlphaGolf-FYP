import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import matplotlib.pyplot as plt


from golf_tournament_dqn import DQNAgent, evaluate_dqn_agent

st.set_page_config(
    page_title="Golf Tournament DQN Predictor",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Golf Tournament Winner Prediction")
st.markdown("### Using Deep Q-Learning to predict golf tournament winners")

# Load pre-processed data
@st.cache_data
def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv") 
    holdout_df = pd.read_csv("holdout_data.csv")
    
    # Load feature list
    with open("feature_list.txt", "r") as f:
        feature_list = [line.strip() for line in f.readlines()]
    
    # Define target columns (not used as features but displayed in UI)
    target_columns = ['hist_winner', 'hist_top3', 'hist_top10', 'hist_top25', 
                     'hist_made_cut', 'position_numeric']
    
    # Filter to only include target columns that exist in the data
    available_target_columns = [col for col in target_columns 
                              if col in train_df.columns or col in test_df.columns or col in holdout_df.columns]
        
    return train_df, test_df, holdout_df, feature_list, available_target_columns

# Load pre-trained model
@st.cache_resource
def load_model(feature_list):
    state_size = len(feature_list)
    agent = DQNAgent(state_size=state_size)
    
    
    try:
        # Try normal loading first
        agent.load_model("./model/dqn_golf_final.h5")
    except (TypeError, ValueError) as e:
        st.warning("Encountered an issue with direct model loading. Building a new model and loading weights.")
        
        # Build the model manually
        agent.model = agent._build_model()
        
        # Try to load just the weights instead
        try:
            agent.model.load_weights("./model/dqn_golf_final.h5")
            st.success("Successfully loaded model weights!")
        except:
            # If that fails too, we'll just use the newly initialized model
            st.error("Could not load model weights. Using a newly initialized model.")
    
    return agent


def main():
    # Show loading message while data loads
    with st.spinner("Loading model and data..."):
        train_df, test_df, holdout_df, feature_list, available_target_columns = load_data()
        agent = load_model(feature_list)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tournament Predictions", 
        "Model Performance", 
        "Player Analysis",
        "About the Model"
    ])
    
    # Tab 1: Tournament Predictions
    with tab1:
        st.header("Tournament Predictions")
        
        # Dataset selector
        dataset_choice = st.radio(
            "Select dataset:",
            ["Test Set", "Holdout Set"],
            horizontal=True
        )
        
        # Use the selected dataset
        if dataset_choice == "Test Set":
            prediction_df = test_df
        else:
            prediction_df = holdout_df
        
        # Tournament selector
        tournaments = prediction_df['tournament_id'].unique()
        selected_tournament = st.selectbox(
            "Select a tournament:",
            tournaments,
            index=0
        )
        
        tournament_data = prediction_df[prediction_df['tournament_id'] == selected_tournament].copy()

        with st.spinner("Generating predictions..."):

            predictions = agent.calculate_win_probability(tournament_data, feature_list)
            sorted_predictions = predictions.sort_values('win_probability', ascending=False).reset_index(drop=True)

            sorted_predictions['rank'] = range(1, len(sorted_predictions) + 1)
        

        st.subheader("Top Predicted Players")
        

        display_cols = ['rank', 'player_id', 'win_probability']
        
        if 'player_name' in sorted_predictions.columns:
            display_cols.insert(2, 'player_name')
        
        for col in available_target_columns:
            if col in sorted_predictions.columns:
                display_cols.append(col)
        
        for col in ['owgr', 'win_percentage']:
            if col in sorted_predictions.columns:
                display_cols.append(col)
        
        
        display_df = sorted_predictions[display_cols].head(10).copy()
        
        # Format probability as percentage
        if 'win_probability' in display_df.columns:
            display_df['win_probability'] = display_df['win_probability'].apply(lambda x: f"{x:.2%}")
            
        # Rename columns for better display
        column_names = {
            'rank': 'Rank',
            'player_id': 'Player ID',
            'player_name': 'Player Name',
            'win_probability': 'Win Probability',
            'owgr': 'World Ranking',
            'win_percentage': 'Historical Win %',
            'hist_winner': 'Winner',
            'hist_top3': 'Top 3',
            'hist_top10': 'Top 10',
            'hist_top25': 'Top 25',
            'hist_made_cut': 'Made Cut',
            'position_numeric': 'Final Position'
        }
        
        display_df = display_df.rename(columns={col: column_names.get(col, col) for col in display_df.columns})
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)
        
        # Find and highlight actual winner
        actual_winner = sorted_predictions[sorted_predictions['is_winner'] == 1]
        if not actual_winner.empty:
            winner_rank = actual_winner.index[0] + 1
            winner_prob = actual_winner['win_probability'].values[0]
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            col1.metric("Actual Winner Rank", f"#{winner_rank}")
            col2.metric("Winner's Predicted Probability", f"{winner_prob:.2%}")
            col3.metric("Top-10 Accuracy", "Yes" if winner_rank <= 10 else "No")
            
            # Success message
            if winner_rank <= 3:
                st.success(f"Successfully predicted the winner in the top 3!")
            elif winner_rank <= 10:
                st.info(f"Actual winner was in our top 10 predictions (#{winner_rank})")
            else:
                st.warning(f"⚠️ Actual winner was ranked #{winner_rank} in our predictions")
                
            # Display winner details if target columns are available
            winner_info = actual_winner[available_target_columns].iloc[0].to_dict() if not actual_winner.empty else {}
            if winner_info and any(winner_info.values()):
                st.subheader("Actual Winner Historical Performance")
                info_cols = st.columns(len(winner_info))
                
                for i, (col, value) in enumerate(winner_info.items()):
                    display_value = "Yes" if value == 1 else ("No" if value == 0 else value)
                    info_cols[i].metric(column_names.get(col, col), display_value)
        
        st.subheader("Win Probability Distribution")
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Get data for top 15 players
        top_n = min(15, len(sorted_predictions))
        ranks = list(range(1, top_n + 1))
        probabilities = sorted_predictions['win_probability'].head(top_n).values
        
        # Create bar chart
        bars = ax.bar(ranks, probabilities, color='cornflowerblue')
        
        # Highlight actual winner if in top 15
        if not actual_winner.empty and winner_rank <= top_n:
            bars[winner_rank - 1].set_color('green')
        
        ax.set_xlabel("Player Rank")
        ax.set_ylabel("Win Probability")
        ax.set_xticks(ranks)
        ax.set_ylim(0, max(probabilities) * 1.1)  # Add some headroom
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels on top of bars
        for i, v in enumerate(probabilities):
            ax.text(i + 1, v + 0.01, f"{v:.1%}", ha='center')
        
        st.pyplot(fig)
    
    with tab2:
        st.header("Model Performance Metrics")
        
        with st.spinner("Calculating performance metrics..."):
            test_metrics, _ = evaluate_dqn_agent(agent, test_df, feature_list)
            holdout_metrics, _ = evaluate_dqn_agent(agent, holdout_df, feature_list)
            
            # Performance metrics in columns
            st.subheader("Winner Prediction Accuracy")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Exact Winner", f"{test_metrics['accuracy']:.1%}")
            col2.metric("Top-3 Accuracy", f"{test_metrics['top3_accuracy']:.1%}")
            col3.metric("Top-5 Accuracy", f"{test_metrics['top5_accuracy']:.1%}")
            col4.metric("Top-10 Accuracy", f"{test_metrics['top10_accuracy']:.1%}")
            
            # Average winner stats
            st.subheader("Winner Prediction Statistics")
            col1, col2 = st.columns(2)
            col1.metric("Avg. Winner Probability", f"{test_metrics['avg_winner_probability']:.1%}")
            col2.metric("Avg. Winner Rank", f"{test_metrics['avg_winner_rank']:.1f}")
            
            # Comparison chart: Test vs Holdout
            st.subheader("Test vs Holdout Performance")
            
            # Data for chart
            n_values = [1, 3, 5, 10]
            test_acc = [test_metrics['accuracy'], test_metrics['top3_accuracy'], 
                       test_metrics['top5_accuracy'], test_metrics['top10_accuracy']]
            holdout_acc = [holdout_metrics['accuracy'], holdout_metrics['top3_accuracy'], 
                          holdout_metrics['top5_accuracy'], holdout_metrics['top10_accuracy']]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(n_values, test_acc, 'o-', label='Test Set')
            ax.plot(n_values, holdout_acc, 's-', label='Holdout Set')
            ax.set_title('Top-N Accuracy Comparison')
            ax.set_xlabel('N (Top-N Prediction)')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(n_values)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            st.pyplot(fig)
            
            # Additional info about model performance
            st.info("""
            **Interpretation:**
            - Exact winner prediction is challenging in golf due to the sport's variability
            - Top-5 and Top-10 accuracy are strong indicators of model quality
            - Comparable performance between test and holdout sets suggests good generalization
            """)
    with tab3:
        st.header("Player Analysis")
        
        # Combine datasets for player analysis
        all_data = pd.concat([train_df, test_df, holdout_df])
        
        # Get unique players
        players = all_data['player_id'].unique()
        
        # If player names are available, use them in the dropdown
        if 'player_name' in all_data.columns:
            player_names = {row['player_id']: row['player_name'] 
                           for _, row in all_data[['player_id', 'player_name']].drop_duplicates().iterrows()}
            player_options = [f"{pid} - {player_names.get(pid, 'Unknown')}" for pid in players]
        else:
            player_options = [str(pid) for pid in players]
        
        # Player selector
        selected_player_option = st.selectbox("Select a player:", player_options)
        selected_player = int(selected_player_option.split(' - ')[0])
        
        # Get player data
        player_data = all_data[all_data['player_id'] == selected_player].copy()
        
        if not player_data.empty:
            # Player statistics
            tournaments_played = player_data['tournament_id'].nunique()
            wins = player_data['is_winner'].sum()
            win_pct = 0 if tournaments_played == 0 else (wins / tournaments_played) * 100
            
            # Display player stats
            st.subheader("Player Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tournaments Played", tournaments_played)
            col2.metric("Wins", wins)
            col3.metric("Win Percentage", f"{win_pct:.1f}%")
            
            # Display target column statistics if available
            target_metrics = {}
            for col in available_target_columns:
                if col in player_data.columns:
                    # Count occurrences of 1 in this column
                    count = player_data[col].sum()
                    pct = 0 if tournaments_played == 0 else (count / tournaments_played) * 100
                    target_metrics[column_names.get(col, col)] = (count, pct)
            
            if target_metrics:
                st.subheader("Historical Performance")
                metric_cols = st.columns(len(target_metrics))
                
                for i, (label, (count, pct)) in enumerate(target_metrics.items()):
                    metric_cols[i].metric(label, count, f"{pct:.1f}%")
            
            # Player win probability distribution across tournaments
            if 'win_probability' in player_data.columns:
                st.subheader("Win Probability Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                player_probs = player_data['win_probability'].dropna()
                
                if not player_probs.empty:
                    ax.hist(player_probs, bins=20, color='skyblue', edgecolor='black')
                    ax.axvline(player_probs.mean(), color='red', linestyle='--', 
                               label=f'Mean: {player_probs.mean():.2%}')
                    ax.set_xlabel("Predicted Win Probability")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.write("No win probability data available for this player")
            else:
                st.write("Win probability data not available")
        else:
            st.warning("No data available for selected player")
    
        st.subheader("DQN Architecture")
        
    
        from PIL import Image
        import io
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    
        fig, ax = plt.subplots(figsize=(10, 6))
        
        input_rect = patches.Rectangle((0, 0), 0.6, 6, linewidth=1, edgecolor='r', facecolor='lightcoral', alpha=0.7)
        ax.add_patch(input_rect)
        ax.text(0.3, 6.2, "Input Layer", ha='center')
        ax.text(0.3, 3, f"({len(feature_list)} features)", ha='center', rotation=90)
        
        # Hidden layers
        h1_rect = patches.Rectangle((2, 0.5), 0.6, 5, linewidth=1, edgecolor='g', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(h1_rect)
        ax.text(2.3, 6.2, "Hidden Layer 1", ha='center')
        ax.text(2.3, 3, "(128 neurons)", ha='center', rotation=90)
        
        h2_rect = patches.Rectangle((4, 1), 0.6, 4, linewidth=1, edgecolor='g', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(h2_rect)
        ax.text(4.3, 6.2, "Hidden Layer 2", ha='center')
        ax.text(4.3, 3, "(64 neurons)", ha='center', rotation=90)
        
        h3_rect = patches.Rectangle((6, 1.5), 0.6, 3, linewidth=1, edgecolor='g', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(h3_rect)
        ax.text(6.3, 6.2, "Hidden Layer 3", ha='center')
        ax.text(6.3, 3, "(32 neurons)", ha='center', rotation=90)
        
        # Output layer
        output_rect = patches.Rectangle((8, 2.5), 0.6, 1, linewidth=1, edgecolor='b', facecolor='lightblue', alpha=0.7)
        ax.add_patch(output_rect)
        ax.text(8.3, 6.2, "Output Layer", ha='center')
        ax.text(8.3, 3, "(2 actions)", ha='center')
        
        # Connecting lines
        for i in range(5):
            ax.plot([0.6, 2], [i+0.5, i+0.5], 'k-', alpha=0.3)
        
        for i in range(4):
            ax.plot([2.6, 4], [i+1, i+1], 'k-', alpha=0.3)
        
        for i in range(3):
            ax.plot([4.6, 6], [i+1.5, i+1.5], 'k-', alpha=0.3)
        
        for i in range(1):
            ax.plot([6.6, 8], [i+2.5, i+2.5], 'k-', alpha=0.3)
        
        # Titles and labels
        ax.text(4.3, 7, "Deep Q-Network Architecture", ha='center', fontsize=14, fontweight='bold')
    
        ax.set_xlim(-1, 10)
        ax.set_ylim(-0.5, 8)
        ax.axis('off')
        
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()