# feature_engineering/processors/player_profile_processor.py

import pandas as pd
import numpy as np
from ..base import BaseProcessor

class PlayerProfileProcessor(BaseProcessor):
    def extract_features(self, player_ids=None, season=None, tournament_id=None):
        profile_df = self.data_extractor.extract_player_profile(player_ids=player_ids)

        if profile_df.empty:
            return pd.DataFrame()

        return self._extract_profile_features(profile_df)

    def _extract_profile_features(self, profile_df):
        features = profile_df.copy()
        owgr_features = self._process_owgr_data(features)
        if not owgr_features.empty:
            features = pd.merge(features, owgr_features, on='player_id', how='left')

        if 'snapshot' in features.columns:
            snapshot_features = features['snapshot'].apply(self._process_snapshot_data)
            snapshot_df = pd.DataFrame(snapshot_features.tolist())
            features = pd.concat([features.drop(columns=['snapshot']), snapshot_df], axis=1)

        return features

    def _process_snapshot_data(self, snapshot_data):
        snapshot_features = {}

        for item in snapshot_data:
            title = item.get("title", "").lower().replace(" ", "_")
            value = item.get("value", "")
            description = item.get("description", "")

            snapshot_features[f"snapshot_{title}"] = value

            if description:
                snapshot_features[f"snapshot_{title}_desc"] = description

            if title == "lowest_round":
                try:
                    score = int(value) if value.isdigit() else None
                    snapshot_features["lowest_round_score"] = score
                except (ValueError, AttributeError):
                    pass

        return snapshot_features


    def _process_owgr_data(self, profile_data):
        if 'player_id' not in profile_data.columns or profile_data.empty:
            return pd.DataFrame()

        owgr_features = profile_data[['player_id']].copy()
        owgr_columns = ['standings_owgr', 'debug_owgr', 'owgr']
        owgr_found = False

        for col in owgr_columns:
            if col in profile_data.columns and not profile_data[col].isna().all():
                owgr_features['owgr'] = pd.to_numeric(profile_data[col], errors='coerce')
                owgr_found = True
                break

        if owgr_found:
            conditions = [
                (owgr_features['owgr'] <= 10),
                (owgr_features['owgr'] <= 25) & (owgr_features['owgr'] > 10),
                (owgr_features['owgr'] <= 50) & (owgr_features['owgr'] > 25),
                (owgr_features['owgr'] <= 100) & (owgr_features['owgr'] > 50),
                (owgr_features['owgr'] > 100)
            ]

            tier_labels = ['Elite', 'Top 25', 'Top 50', 'Top 100', 'Outside 100']
            owgr_features['owgr_tier'] = np.select(conditions, tier_labels, default='Unknown')

            owgr_features['owgr_score'] = 1000 - (100 * np.log10(owgr_features['owgr']))

            if len(owgr_features) > 1:
                max_score = owgr_features['owgr_score'].max()
                min_score = owgr_features['owgr_score'].min()
                if max_score > min_score:
                    owgr_features['owgr_score_norm'] = (
                        (owgr_features['owgr_score'] - min_score) / (max_score - min_score) * 100
                    )
            else:
                owgr_features['owgr_score_norm'] = 100 - (owgr_features['owgr'] / 200 * 100).clip(0, 100)

        return owgr_features
