import pandas as pd
import numpy as np
from ..base import BaseProcessor

class PlayerCareerProcessor(BaseProcessor):    
    def extract_features(self, player_ids=None, season=None, tournament_id=None):

        career_df = self.data_extractor.extract_player_career(player_ids=player_ids)

        yearly_df = self.data_extractor.extract_player_career_yearly(
            player_ids=player_ids,
            years=season
        )
        
        if career_df.empty and yearly_df.empty:
            return pd.DataFrame()

        features = self._combine_career_data(career_df, yearly_df, season)
        
        return features
    
    def _combine_career_data(self, career_df, yearly_df, season=None):
        if career_df.empty:
            return pd.DataFrame()
        
        features = career_df.copy()

        if not yearly_df.empty:
            if season:
                recent_perf = yearly_df[yearly_df['year'] == season]
            else:

                recent_perf = yearly_df.sort_values(['player_id', 'year'], ascending=[True, False])
                recent_perf = recent_perf.groupby('player_id').first().reset_index()

            perf_cols = ['player_id']
            stat_cols = [
                'events', 'wins', 'top10', 'top25', 'cuts_made',
                'official_money'
            ]
            perf_cols.extend([col for col in stat_cols if col in recent_perf.columns])
            
            features = pd.merge(
                features,
                recent_perf[perf_cols],
                on='player_id',
                how='left',
                suffixes=('', '_current')
            )

            if 'cuts_made_current' in features.columns and 'events_current' in features.columns:
                features['current_cut_pct'] = features['cuts_made_current'] / features['events_current']
            
            if 'top10_current' in features.columns and 'events_current' in features.columns:
                features['current_top10_pct'] = features['top10_current'] / features['events_current']
            
            if 'top25_current' in features.columns and 'events_current' in features.columns:
                features['current_top25_pct'] = features['top25_current'] / features['events_current']

        if not yearly_df.empty:
            career_stats = yearly_df.groupby('player_id').agg({
                'events': 'sum',
                'wins': 'sum',
                'top10': 'sum',
                'top25': 'sum',
                'cuts_made': 'sum',
                'official_money': 'sum'
            }).reset_index()

            career_stats = career_stats.rename(columns={
                col: f'career_{col}' for col in career_stats.columns if col != 'player_id'
            })
            
            features = pd.merge(
                features,
                career_stats,
                on='player_id',
                how='left'
            )
            
            if 'career_cuts_made' in features.columns and 'career_events' in features.columns:
                features['career_cut_pct'] = features['career_cuts_made'] / features['career_events']
            
            if 'career_top10' in features.columns and 'career_events' in features.columns:
                features['career_top10_pct'] = features['career_top10'] / features['career_events']
            
            if 'career_top25' in features.columns and 'career_events' in features.columns:
                features['career_top25_pct'] = features['career_top25'] / features['career_events']
 
        features = self._process_achievement_data(features)
        
        return features
    
    def _process_achievement_data(self, df):
        if df.empty:
            return df

        df_processed = df.copy()

        money_cols = [col for col in df_processed.columns if 'money' in col.lower() or 'earnings' in col.lower()]
        for col in money_cols:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].replace('-', None)
                df_processed[col] = df_processed[col].astype(str).str.replace('[\\$,]', '', regex=True)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        fraction_cols = [col for col in df_processed.columns if df_processed[col].astype(str).str.contains('/').any()]
        for col in fraction_cols:
            if col in df_processed.columns:
                fractions = df_processed[col].astype(str).str.split('/', expand=True)
                if len(fractions.columns) == 2:
                    df_processed[f"{col}_numerator"] = pd.to_numeric(fractions[0], errors='coerce')
                    df_processed[f"{col}_denominator"] = pd.to_numeric(fractions[1], errors='coerce')
                    df_processed[f"{col}_pct"] = df_processed[f"{col}_numerator"] / df_processed[f"{col}_denominator"]
        if 'career_cut_pct' in df_processed.columns and 'current_cut_pct' in df_processed.columns:
            df_processed['cut_consistency'] = df_processed['current_cut_pct'] - df_processed['career_cut_pct']
        
        if 'career_top10_pct' in df_processed.columns and 'current_top10_pct' in df_processed.columns:
            df_processed['top10_consistency'] = df_processed['current_top10_pct'] - df_processed['career_top10_pct']
        
        if 'career_top25_pct' in df_processed.columns and 'current_top25_pct' in df_processed.columns:
            df_processed['top25_consistency'] = df_processed['current_top25_pct'] - df_processed['career_top25_pct']
        
        return df_processed