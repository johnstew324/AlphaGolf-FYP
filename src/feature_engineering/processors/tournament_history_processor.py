import pandas as pd
import numpy as np
from ..base import BaseProcessor

class TournamentHistoryProcessor(BaseProcessor):
    
    def extract_features(self, tournament_id=None, player_ids=None, season=None):
        get_player_features = player_ids is not None
        
        if get_player_features:
            return self._get_player_tournament_features(tournament_id, player_ids, season)
        else:
            return self._get_tournament_features(tournament_id, season)
    
    def _get_player_tournament_features(self, tournament_id, player_ids, season):
        history_df = self.data_extractor.extract_tournament_history(
            tournament_ids=tournament_id,
            player_ids=player_ids,
            years=season
        )
        
        if history_df.empty:
            return pd.DataFrame()
        
        features = self._process_player_history(history_df)
        
        return features
    
    def _get_tournament_features(self, tournament_id, season):
        history_df = self.data_extractor.extract_tournament_history(
            tournament_ids=tournament_id,
            years=season
        )
        
        if history_df.empty:
            return pd.DataFrame()

        features = self._process_tournament_history(history_df)
        
        return features
    
    def _process_player_history(self, history_df):
        df = history_df.copy()
        df['position_numeric'] = df['position'].apply(
            lambda x: pd.to_numeric(x.replace('T', ''), errors='coerce') if isinstance(x, str) else x
        )

        features = pd.DataFrame()

        if not df.empty and 'player_id' in df.columns:
            group_cols = ['player_id']
            if 'tournament_id' in df.columns:
                group_cols.append('tournament_id')

            grouped = df.groupby(group_cols)

            features = grouped.agg({
                'position_numeric': ['count', 'mean', 'min', 'max', 'std'],
                'year': ['min', 'max']
            }).reset_index()

            features.columns = ['_'.join(col).strip('_') for col in features.columns.values]

            column_map = {
                'player_id_': 'player_id',
                'tournament_id_': 'tournament_id',
                'position_numeric_count': 'appearances',
                'position_numeric_mean': 'avg_finish',
                'position_numeric_min': 'best_finish',
                'position_numeric_max': 'worst_finish',
                'position_numeric_std': 'finish_std',
                'year_min': 'first_year_played',
                'year_max': 'last_year_played'
            }

            features = features.rename(columns=column_map)

            if all(col in df.columns for col in ['round1_score', 'round2_score', 'round3_score', 'round4_score']):
                round_cols = [f'round{i}_score' for i in range(1, 5)]
                df['rounds_played'] = df[round_cols].notna().sum(axis=1)
                round_stats = df.groupby(group_cols)['rounds_played'].agg(['mean', 'min', 'max']).reset_index()
                round_stats.columns = group_cols + ['avg_rounds_played', 'min_rounds_played', 'max_rounds_played']
                features = pd.merge(features, round_stats, on=group_cols, how='left')

        return features

    
    def _process_tournament_history(self, history_df):
        df = history_df.copy()

        features = pd.DataFrame()
        
        if not df.empty:

            grouped = df.groupby('tournament_id')
            
            features = grouped.agg({
                'year': ['count', 'min', 'max'],
                'winning_score_to_par': ['mean', 'min', 'max', 'std'],
                'player_count': ['mean', 'min', 'max']
            }).reset_index()

            features.columns = ['_'.join(col).strip('_') for col in features.columns.values]

            column_map = {
                'tournament_id_': 'tournament_id',
                'year_count': 'years_recorded',
                'year_min': 'first_year',
                'year_max': 'last_year',
                'winning_score_to_par_mean': 'avg_winning_score',
                'winning_score_to_par_min': 'best_winning_score',
                'winning_score_to_par_max': 'worst_winning_score',
                'winning_score_to_par_std': 'winning_score_std',
                'player_count_mean': 'avg_field_size',
                'player_count_min': 'min_field_size',
                'player_count_max': 'max_field_size'
            }
            
            features = features.rename(columns=column_map)
            features['years_span'] = features['last_year'] - features['first_year']
            features['score_variability'] = features['winning_score_std'] / features['avg_winning_score']
            
            winners = df.groupby(['tournament_id', 'winner_name']).size().reset_index(name='wins')
            top_winners = winners.sort_values(['tournament_id', 'wins'], ascending=[True, False])
            top_winners = top_winners.groupby('tournament_id').head(3)

            top_winners['rank'] = top_winners.groupby('tournament_id').cumcount() + 1
            top_winners_pivot = top_winners.pivot(
                index='tournament_id', 
                columns='rank', 
                values=['winner_name', 'wins']
            )
    
            top_winners_pivot.columns = [
                f'{col[0]}_{col[1]}' for col in top_winners_pivot.columns
            ]
            top_winners_pivot = top_winners_pivot.reset_index()
            
            features = pd.merge(features, top_winners_pivot, on='tournament_id', how='left')
        
        return features



    def extract_position_and_winner_data(data_extractor, tournament_id, player_ids):
        print(f"Extracting position and winner data for tournament {tournament_id}")
        
    
        tournament_history = data_extractor.extract_tournament_history(
            tournament_ids=tournament_id,
            player_ids=player_ids
        )
        
        if tournament_history.empty:
            print(f"No tournament history found for {tournament_id}")
            return pd.DataFrame()
        

        position_data = []
        
        for _, player_data in tournament_history.iterrows():
            player_info = {
                'player_id': player_data['player_id'],
                'tournament_id': tournament_id
            }
            
            if 'position' in player_data:
                player_info['position'] = player_data['position']
                
                if isinstance(player_data['position'], str):
                    numeric_position = player_data['position'].replace('T', '')
                    try:
                        numeric_position = int(numeric_position)
                        player_info['position_numeric'] = numeric_position
                        
                        player_info['is_winner'] = 1 if numeric_position == 1 else 0
                        player_info['is_top3'] = 1 if numeric_position <= 3 else 0
                        player_info['is_top10'] = 1 if numeric_position <= 10 else 0
                        player_info['is_top25'] = 1 if numeric_position <= 25 else 0
                    except:

                        player_info['position_numeric'] = None
                        player_info['is_winner'] = 0
                        player_info['is_top3'] = 0
                        player_info['is_top10'] = 0
                        player_info['is_top25'] = 0
                else:
                    player_info['position_numeric'] = player_data.get('position_numeric')
                    position_value = player_info['position_numeric']
                    
                    if pd.notna(position_value):
                        player_info['is_winner'] = 1 if position_value == 1 else 0
                        player_info['is_top3'] = 1 if position_value <= 3 else 0
                        player_info['is_top10'] = 1 if position_value <= 10 else 0
                        player_info['is_top25'] = 1 if position_value <= 25 else 0
                    else:
                        player_info['is_winner'] = 0
                        player_info['is_top3'] = 0
                        player_info['is_top10'] = 0
                        player_info['is_top25'] = 0
            
            if 'score_to_par' in player_data:
                player_info['score_to_par'] = player_data['score_to_par']
            
            if 'total_score' in player_data:
                player_info['total_score'] = player_data['total_score']
                
            position_data.append(player_info)
        
        position_df = pd.DataFrame(position_data)
        
        if not position_df.empty and 'is_winner' in position_df.columns:
            winners = position_df[position_df['is_winner'] == 1]
            if not winners.empty:
                winner_id = winners['player_id'].iloc[0]
                # Add winner_id as a column to all rows
                position_df['tournament_winner_id'] = winner_id
        
        return position_df