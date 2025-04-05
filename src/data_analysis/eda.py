import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GolfDataAnalyzer:
    def __init__(self, uri: str = None, database_name: str = "pga_tour_data"):
    
        if uri is None:
            uri = os.getenv('MONGODB_URI')
            
        if uri is None:
            raise ValueError("MongoDB URI not provided and not found in environment variables")
            
        self.client = MongoClient(uri)
        self.db = self.client[database_name]
        self.results_dir = Path("AlphaGolf\data_analysis_results") 
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Connected to database: {database_name}")
        
    def load_tournament_history(self, min_year: int = 2019, limit: int = None) -> pd.DataFrame:
        logger.info(f"Loading tournament history from {min_year} onwards")
        
        query = {"year": {"$gte": min_year}}
        projection = {
            "tournament_id": 1,
            "year": 1,
            "players": 1,
            "winner": 1
        }
        
        cursor = self.db.tournament_history.find(query, projection).sort("year", -1)
        if limit:
            cursor = cursor.limit(limit)
            
        tournaments = list(cursor)
        
        if not tournaments:
            logger.warning("No tournament history found")
            return pd.DataFrame()
            
        logger.info(f"Loaded {len(tournaments)} tournaments")
        
        # Flatten player results data
        results = []
        for tournament in tournaments:
            tournament_id = tournament.get("tournament_id")
            year = tournament.get("year")
            
            for player in tournament.get("players", []):
                player_result = {
                    "tournament_id": tournament_id,
                    "year": year,
                    "player_id": player.get("player_id"),
                    "player_name": player.get("name"),
                    "position": player.get("position"),
                    "total_score": player.get("total_score"),
                    "par_relative": player.get("par_relative")
                }
                
                # Add round scores if available
                for i, round_data in enumerate(player.get("rounds", [])):
                    player_result[f"round{i+1}_score"] = round_data.get("score")
                    player_result[f"round{i+1}_par_relative"] = round_data.get("par_relative")
                
                results.append(player_result)
        
        df = pd.DataFrame(results)
        
        # Process position strings (e.g., "T17" -> 17)
        if "position" in df.columns:
            df["position_numeric"] = df["position"].apply(self._parse_position)
            
        logger.info(f"Created DataFrame with {len(df)} player-tournament results")
        return df
    
    def _parse_position(self, position: str) -> Optional[int]:
        """Parse position string to numeric value"""
        if pd.isna(position) or position is None:
            return None
            
        # Handle "CUT", "WD", etc.
        if not isinstance(position, str) or not any(c.isdigit() for c in position):
            return None
            
        # Remove 'T' prefix for tied positions
        position = position.replace("T", "")
        
        try:
            return int(position)
        except ValueError:
            return None
    
    def load_player_stats(self, seasons: List[int] = None) -> pd.DataFrame:
        query = {}
        if seasons:
            query["season"] = {"$in": seasons}
            
        cursor = self.db.player_stats.find(query)
        player_stats = list(cursor)
        
        if not player_stats:
            logger.warning("No player stats found")
            return pd.DataFrame()
            
        logger.info(f"Loaded stats for {len(player_stats)} player-seasons")
        
        return pd.DataFrame(player_stats)
    
    def load_strokes_gained_data(self, min_season: int = 2014) -> pd.DataFrame:
        logger.info(f"Loading strokes gained data from {min_season} onwards")
        
        # Find player stats documents that have strokes gained data
        # Note: In your data structure, the category has a comma in it
        query = {
            "season": {"$gte": min_season},
            "$or": [
                {"stats.STROKES_GAINED, SCORING": {"$exists": True}},
                {"stats.STROKES_GAINED, DRIVING": {"$exists": True}},
                {"stats.STROKES_GAINED, APPROACH": {"$exists": True}},
                {"stats.STROKES_GAINED, AROUND_GREEN": {"$exists": True}},
                {"stats.STROKES_GAINED, PUTTING": {"$exists": True}}
            ]
        }
        
        player_stats = list(self.db.player_stats.find(query))
        
        if not player_stats:
            logger.warning("No strokes gained data found")
            return pd.DataFrame()
        sg_data = []
        
        for player_doc in player_stats:
            player_id = player_doc.get("player_id")
            player_name = player_doc.get("name")
            season = player_doc.get("season")
            stats = player_doc.get("stats", {})
            
            player_sg = {
                "player_id": player_id,
                "player_name": player_name,
                "season": season
            }
            sg_categories = [
                "STROKES_GAINED, SCORING",
                "STROKES_GAINED, DRIVING",
                "STROKES_GAINED, APPROACH",
                "STROKES_GAINED, AROUND_GREEN",
                "STROKES_GAINED, PUTTING"
            ]
            
            for category in sg_categories:
                if category in stats:
                    for stat in stats[category]:
                        stat_name = stat.get("title", "").replace("SG: ", "sg_").lower().replace(" ", "_").replace("-", "_")
                        try:
                            player_sg[stat_name] = float(stat.get("value", 0))
                        except (ValueError, TypeError):
                            player_sg[stat_name] = None
                        
                        player_sg[f"{stat_name}_rank"] = stat.get("rank")
            
            sg_data.append(player_sg)
        
        df = pd.DataFrame(sg_data)
        logger.info(f"Created DataFrame with strokes gained data for {len(df)} player-seasons")
        
        return df
    
    def load_course_fit_data(self) -> pd.DataFrame:
        logger.info("Loading course fit data")
        
        query = {"field_stat_type": "COURSE_FIT"}
        course_fit_docs = list(self.db.field_stats.find(query))
        
        if not course_fit_docs:
            logger.warning("No course fit data found")
            return pd.DataFrame()
            
        fit_data = []
        
        for doc in course_fit_docs:
            tournament_id = doc.get("tournament_id")
            collected_at = doc.get("collected_at")
            
            for player in doc.get("players", []):
                player_id = player.get("player_id")
                score = player.get("score")
                total_rounds = player.get("total_rounds")
                
                player_data = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "course_fit_score": score,
                    "total_rounds": total_rounds,
                    "collected_at": collected_at
                }
                
                # Extract specific stat values
                for stat in player.get("stats", []):
                    header = stat.get("header", "")
                    if not header:
                        continue
                        
                    # Clean stat name for column name
                    stat_name = header.lower().replace(" ", "_").replace(":", "").replace("-", "_")
                    
                    # Add value and rank
                    player_data[f"{stat_name}_value"] = stat.get("value")
                    player_data[f"{stat_name}_rank"] = stat.get("rank")
                
                fit_data.append(player_data)
        
        df = pd.DataFrame(fit_data)
        logger.info(f"Created DataFrame with course fit data for {len(df)} player-tournament pairs")
        
        return df
    
    def load_current_form_data(self) -> pd.DataFrame:
        logger.info("Loading current form data")
        
        form_docs = list(self.db.current_form.find({}))
        
        if not form_docs:
            logger.warning("No current form data found")
            return pd.DataFrame()
        form_data = []
        
        for doc in form_docs:
            tournament_id = doc.get("tournament_id")
            collected_at = doc.get("collected_at")
            
            for player in doc.get("players", []):
                player_id = player.get("player_id")
                total_rounds = player.get("total_rounds", 0)
                
                player_form = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "total_rounds": total_rounds,
                    "collected_at": collected_at
                }
                
                # Process recent tournament results - extract only numeric values
                recent_results = player.get("tournament_results", [])
                for i, result in enumerate(recent_results[:5]):  # Take up to 5 most recent
                    # Skip storing names and other string fields
                    player_form[f"recent_tournament_{i+1}_score"] = self._safe_convert_to_numeric(result.get("score"))
                    # Parse position to numeric
                    player_form[f"recent_tournament_{i+1}_position_numeric"] = self._parse_position(result.get("position"))
                
                # Process strokes gained stats
                sg_stats = player.get("strokes_gained", [])
                for stat in sg_stats:
                    stat_id = stat.get("stat_id")
                    stat_value = self._safe_convert_to_numeric(stat.get("stat_value"))
                    
                    if stat_id == "02567":  # SG: OTT
                        player_form["recent_sg_ott"] = stat_value
                    elif stat_id == "02568":  # SG: APP
                        player_form["recent_sg_app"] = stat_value
                    elif stat_id == "02569":  # SG: ATG
                        player_form["recent_sg_atg"] = stat_value
                    elif stat_id == "02564":  # SG: P
                        player_form["recent_sg_p"] = stat_value
                    elif stat_id == "02575":  # SG: TOT
                        player_form["recent_sg_tot"] = stat_value
                
                form_data.append(player_form)
        
        df = pd.DataFrame(form_data)
        
        position_cols = [col for col in df.columns if 'recent_tournament_' in col and 'position_numeric' in col]
        if position_cols:
            df['avg_recent_position'] = df[position_cols].mean(axis=1)
        
        logger.info(f"Created DataFrame with current form data for {len(df)} player-tournament pairs")
        
        return df
    
    def _safe_convert_to_numeric(self, value) -> Optional[float]:
        """Safely convert a value to numeric, returning None if not possible"""
        if pd.isna(value) or value is None:
            return None
            
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def load_tournament_history_stats(self) -> pd.DataFrame:
        logger.info("Loading tournament history statistics")
        
        hist_docs = list(self.db.tournament_history_stats.find({}))
        
        if not hist_docs:
            logger.warning("No tournament history statistics found")
            return pd.DataFrame()
            
        # Extract and flatten history data
        history_data = []
        
        for doc in hist_docs:
            tournament_id = doc.get("tournament_id")
            collected_at = doc.get("collected_at")
            
            for player in doc.get("players", []):
                player_id = player.get("player_id")
                total_rounds = player.get("total_rounds", 0)
                
                player_history = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "total_rounds": total_rounds,
                    "collected_at": collected_at
                }
                historical_results = player.get("tournament_results", [])
                positions = []
                
                for i, result in enumerate(historical_results):
                    score = self._safe_convert_to_numeric(result.get("score"))
                    if score is not None:
                        player_history[f"hist_result_{i+1}_score"] = score
                    
                    pos_numeric = self._parse_position(result.get("position"))
                    if pos_numeric:
                        positions.append(pos_numeric)
                
                if positions:
                    player_history["avg_historical_position"] = sum(positions) / len(positions)
                    player_history["best_historical_position"] = min(positions)
                    player_history["historical_appearances"] = len(positions)
                
                sg_stats = player.get("strokes_gained", [])
                for stat in sg_stats:
                    stat_id = stat.get("stat_id")
                    stat_value = self._safe_convert_to_numeric(stat.get("stat_value"))
                    
                    if stat_value is not None:
                        if stat_id == "02567":  
                            player_history["historical_sg_ott"] = stat_value
                        elif stat_id == "02568":  
                            player_history["historical_sg_app"] = stat_value
                        elif stat_id == "02569":  
                            player_history["historical_sg_atg"] = stat_value
                        elif stat_id == "02564": 
                            player_history["historical_sg_p"] = stat_value
                        elif stat_id == "02575":
                            player_history["historical_sg_tot"] = stat_value
                
                history_data.append(player_history)
        
        df = pd.DataFrame(history_data)
        logger.info(f"Created DataFrame with tournament history stats for {len(df)} player-tournament pairs")
        
        return df
    
    def load_tournament_weather(self) -> pd.DataFrame:
        logger.info("Loading tournament weather data")
        
        weather_docs = list(self.db.tournament_weather.find({}))
        
        if not weather_docs:
            logger.warning("No tournament weather data found")
            return pd.DataFrame()
            
        weather_data = []
        
        for doc in weather_docs:
            tournament_id = doc.get("tournament_id")
            year = doc.get("year")
            
            tournament_weather = {
                "tournament_id": tournament_id,
                "year": year
            }
            
            for i, round_data in enumerate(doc.get("rounds", [])):
                tournament_weather[f"round{i+1}_temp"] = self._safe_convert_to_numeric(round_data.get("temp"))
                tournament_weather[f"round{i+1}_windspeed"] = self._safe_convert_to_numeric(round_data.get("windspeed"))
                tournament_weather[f"round{i+1}_windgust"] = self._safe_convert_to_numeric(round_data.get("windgust"))
                tournament_weather[f"round{i+1}_winddir"] = self._safe_convert_to_numeric(round_data.get("winddir"))
                tournament_weather[f"round{i+1}_precip"] = self._safe_convert_to_numeric(round_data.get("precip"))
                tournament_weather[f"round{i+1}_humidity"] = self._safe_convert_to_numeric(round_data.get("humidity"))
            
            temp_cols = [col for col in tournament_weather if 'temp' in col]
            wind_cols = [col for col in tournament_weather if 'windspeed' in col]
            
            temp_values = [tournament_weather[col] for col in temp_cols 
                          if col in tournament_weather and tournament_weather[col] is not None]
            wind_values = [tournament_weather[col] for col in wind_cols 
                           if col in tournament_weather and tournament_weather[col] is not None]
            
            if temp_values:
                tournament_weather["avg_temp"] = np.mean(temp_values)
            if wind_values:
                tournament_weather["avg_windspeed"] = np.mean(wind_values)
            
            weather_data.append(tournament_weather)
        
        df = pd.DataFrame(weather_data)
        logger.info(f"Created DataFrame with weather data for {len(df)} tournaments")
        
        return df
    
    def create_consolidated_dataset(self) -> pd.DataFrame:
        logger.info("Creating consolidated dataset for modeling")
    
        tournament_results = self.load_tournament_history(min_year=2019)
        if tournament_results.empty:
            logger.error("No tournament results found - cannot create consolidated dataset")
            return pd.DataFrame()
        

        sg_data = self.load_strokes_gained_data()
        
        course_fit = self.load_course_fit_data()
        
        current_form = self.load_current_form_data()
        
        tournament_history = self.load_tournament_history_stats()
        
        weather_data = self.load_tournament_weather()
        
        consolidated = tournament_results.copy()
        

        if not sg_data.empty:
            if 'season' in consolidated.columns:
                consolidated = pd.merge(
                    consolidated,
                    sg_data,
                    how='left',
                    on=['player_id', 'season']
                )
            else:
                consolidated = pd.merge(
                    consolidated,
                    sg_data,
                    how='left',
                    left_on=['player_id', 'year'],
                    right_on=['player_id', 'season']
                )
        
        if not course_fit.empty:
            numeric_cols = course_fit.select_dtypes(include=[np.number]).columns.tolist()
            key_cols = ['tournament_id', 'player_id', 'course_fit_score']
            selected_cols = list(set(numeric_cols + key_cols))

            course_fit_filtered = course_fit[selected_cols].copy()
            
            consolidated = pd.merge(
                consolidated,
                course_fit_filtered,
                how='left',
                on=['tournament_id', 'player_id']
            )
        

        if not current_form.empty:
            numeric_cols = current_form.select_dtypes(include=[np.number]).columns.tolist()
            key_cols = ['tournament_id', 'player_id']
            selected_cols = list(set(numeric_cols + key_cols))
            
            current_form_filtered = current_form[selected_cols].copy()
            
            consolidated = pd.merge(
                consolidated,
                current_form_filtered,
                how='left',
                on=['tournament_id', 'player_id']
            )
        
        if not tournament_history.empty:
            numeric_cols = tournament_history.select_dtypes(include=[np.number]).columns.tolist()
            key_cols = ['tournament_id', 'player_id']
            selected_cols = list(set(numeric_cols + key_cols))
            
            tournament_history_filtered = tournament_history[selected_cols].copy()
            
            consolidated = pd.merge(
                consolidated,
                tournament_history_filtered,
                how='left',
                on=['tournament_id', 'player_id']
            )
        
        if not weather_data.empty:
            numeric_cols = weather_data.select_dtypes(include=[np.number]).columns.tolist()
            key_cols = ['tournament_id', 'year']
            selected_cols = list(set(numeric_cols + key_cols))
            
            weather_data_filtered = weather_data[selected_cols].copy()
            
            consolidated = pd.merge(
                consolidated,
                weather_data_filtered,
                how='left',
                on=['tournament_id', 'year']
            )
        
        logger.info(f"Created consolidated dataset with {len(consolidated)} rows and {len(consolidated.columns)} columns")
        
        return consolidated
    
    def analyze_tournament_outcomes(self, save_plots: bool = True) -> None:
        logger.info("Analyzing tournament outcomes")
        df = self.load_tournament_history(min_year=2019)
        if df.empty:
            logger.warning("No tournament data found for analysis")
            return
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        winning_scores = []
        
        for tournament_id in df['tournament_id'].unique():
            tournament_df = df[df['tournament_id'] == tournament_id]
            winning_row = tournament_df.loc[tournament_df['position_numeric'].idxmin()] if not tournament_df.empty else None
            if winning_row is not None and 'par_relative' in winning_row:
                winning_scores.append((tournament_id, winning_row['par_relative']))
        
        if winning_scores:
            winning_df = pd.DataFrame(winning_scores, columns=['tournament_id', 'winning_score'])
            sns.histplot(data=winning_df, x='winning_score', bins=20, ax=ax1)
            ax1.set_title('Distribution of Winning Scores Relative to Par', fontsize=14)
            ax1.set_xlabel('Score to Par', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            
            mean_score = winning_df['winning_score'].mean()
            ax1.axvline(x=mean_score, color='red', linestyle='--')
            ax1.text(mean_score + 0.5, ax1.get_ylim()[1] * 0.9, f'Mean: {mean_score:.1f}', color='red')
    
        ax2 = fig.add_subplot(gs[0, 1])
        position_groups = []
        for pos in [1, 5, 10, 20, 'CUT']:
            if pos == 'CUT':
                position_filter = df['position'] == 'CUT'
            else:
                position_filter = df['position_numeric'] == pos
            
            scores = df[position_filter]['par_relative'].dropna().tolist()
            if scores:
                for score in scores:
                    position_groups.append(('Position ' + str(pos), score))
        
        if position_groups:
            position_df = pd.DataFrame(position_groups, columns=['position_group', 'score'])
            sns.boxplot(data=position_df, x='position_group', y='score', ax=ax2)
            ax2.set_title('Score Distribution by Position', fontsize=14)
            ax2.set_xlabel('Position', fontsize=12)
            ax2.set_ylabel('Score to Par', fontsize=12)
    
        ax3 = fig.add_subplot(gs[1, 0])
        
        win_margins = []
        for tournament_id in df['tournament_id'].unique():
            tournament_df = df[df['tournament_id'] == tournament_id].copy()
            tournament_df = tournament_df.sort_values('position_numeric')
            
            if len(tournament_df) >= 2:
                winner = tournament_df.iloc[0]
                runner_up = tournament_df.iloc[1]
                
                if 'par_relative' in winner and 'par_relative' in runner_up:
                    win_margin = runner_up['par_relative'] - winner['par_relative']
                    win_margins.append((tournament_id, win_margin))
        
        if win_margins:
            margin_df = pd.DataFrame(win_margins, columns=['tournament_id', 'margin'])
            sns.histplot(data=margin_df, x='margin', bins=10, ax=ax3)
            ax3.set_title('Distribution of Winning Margins', fontsize=14)
            ax3.set_xlabel('Margin (Strokes)', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            
            mean_margin = margin_df['margin'].mean()
            ax3.axvline(x=mean_margin, color='red', linestyle='--')
            ax3.text(mean_margin + 0.1, ax3.get_ylim()[1] * 0.9, f'Mean: {mean_margin:.1f}', color='red')
        
        ax4 = fig.add_subplot(gs[1, 1])
        
        cut_lines = []
        for tournament_id in df['tournament_id'].unique():
            tournament_df = df[df['tournament_id'] == tournament_id].copy()
            made_cut = tournament_df[~tournament_df['position'].isin(['CUT', 'WD', 'DQ'])]
            missed_cut = tournament_df[tournament_df['position'] == 'CUT']
            
            if not made_cut.empty and not missed_cut.empty:
                worst_made_cut = made_cut['par_relative'].max()
                cut_lines.append((tournament_id, worst_made_cut))
        
        if cut_lines:
            cut_df = pd.DataFrame(cut_lines, columns=['tournament_id', 'cut_line'])
            sns.histplot(data=cut_df, x='cut_line', bins=15, ax=ax4)
            ax4.set_title('Distribution of Cut Lines', fontsize=14)
            ax4.set_xlabel('Cut Line (Score to Par)', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)

            mean_cut = cut_df['cut_line'].mean()
            ax4.axvline(x=mean_cut, color='red', linestyle='--')
            ax4.text(mean_cut + 0.5, ax4.get_ylim()[1] * 0.9, f'Mean: {mean_cut:.1f}', color='red')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'tournament_outcomes_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Saved tournament outcomes analysis plot")
        
        plt.close()
    
    def analyze_performance_correlations(self, save_plots: bool = True) -> None:
        logger.info("Analyzing performance correlations")

        df = self.create_consolidated_dataset()
        if df.empty:
            logger.warning("No data found for correlation analysis")
            return
        
        sg_cols = [col for col in df.columns if col.startswith('sg_') and not col.endswith('_rank')]
        course_fit_cols = [col for col in df.columns if 'course_fit' in col]
        recent_form_cols = [col for col in df.columns if 'recent_' in col and not col.endswith('_id')]
        historical_cols = [col for col in df.columns if 'historical_' in col or 'hist_' in col]

        if 'position_numeric' not in df.columns:
            df['position_numeric'] = df['position'].apply(self._parse_position)
        
        position_filter = df['position_numeric'].notna()

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)
        
        if sg_cols:
            ax1 = fig.add_subplot(gs[0, 0])
            
            correlations = []
            for col in sg_cols:
                if col in df.columns:
                    valid_df = df[position_filter & df[col].notna()]
                    
                    if len(valid_df) > 10: 
                        corr = valid_df[['position_numeric', col]].corr().iloc[0, 1]
                        correlations.append((col, corr))
            
            if correlations:
                corr_df = pd.DataFrame(correlations, columns=['metric', 'correlation'])
                corr_df = corr_df.sort_values('correlation')
                sns.barplot(data=corr_df, x='correlation', y='metric', ax=ax1)
                ax1.set_title('Correlation with Tournament Position\n(Negative is Better)', fontsize=14)
                ax1.set_xlabel('Correlation Coefficient', fontsize=12)
                ax1.axvline(x=0, color='black', linestyle='-')
                for i, bar in enumerate(ax1.patches):
                    if bar.get_width() < 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                        
        if course_fit_cols:
            ax2 = fig.add_subplot(gs[0, 1])

            correlations = []
            for col in course_fit_cols:
                if col in df.columns:
                    valid_df = df[position_filter & df[col].notna()]
                    
                    if len(valid_df) > 10: 
                        corr = valid_df[['position_numeric', col]].corr().iloc[0, 1]
                        correlations.append((col, corr))
            
            if correlations:
                corr_df = pd.DataFrame(correlations, columns=['metric', 'correlation'])
                corr_df = corr_df.sort_values('correlation')
                
                sns.barplot(data=corr_df, x='correlation', y='metric', ax=ax2)
                ax2.set_title('Course Fit Metrics Correlation with Position\n(Negative is Better)', fontsize=14)
                ax2.set_xlabel('Correlation Coefficient', fontsize=12)
                ax2.axvline(x=0, color='black', linestyle='-')

                for i, bar in enumerate(ax2.patches):
                    if bar.get_width() < 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
        if recent_form_cols:
            ax3 = fig.add_subplot(gs[1, 0])
            
            correlations = []
            for col in recent_form_cols:
                if col in df.columns:
                    valid_df = df[position_filter & df[col].notna()]
                    
                    if len(valid_df) > 10: 
                        corr = valid_df[['position_numeric', col]].corr().iloc[0, 1]
                        correlations.append((col, corr))
            
            if correlations:
                corr_df = pd.DataFrame(correlations, columns=['metric', 'correlation'])
                corr_df = corr_df.sort_values('correlation')
                
                sns.barplot(data=corr_df, x='correlation', y='metric', ax=ax3)
                ax3.set_title('Recent Form Correlation with Position\n(Negative is Better)', fontsize=14)
                ax3.set_xlabel('Correlation Coefficient', fontsize=12)
                ax3.axvline(x=0, color='black', linestyle='-')
                for i, bar in enumerate(ax3.patches):
                    if bar.get_width() < 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
        if historical_cols:
            ax4 = fig.add_subplot(gs[1, 1])
            
            correlations = []
            for col in historical_cols:
                if col in df.columns:
                    valid_df = df[position_filter & df[col].notna()]
                    
                    if len(valid_df) > 10: 
                        corr = valid_df[['position_numeric', col]].corr().iloc[0, 1]
                        correlations.append((col, corr))
            
            if correlations:
                corr_df = pd.DataFrame(correlations, columns=['metric', 'correlation'])
                corr_df = corr_df.sort_values('correlation')
                
                sns.barplot(data=corr_df, x='correlation', y='metric', ax=ax4)
                ax4.set_title('Historical Performance Correlation with Position\n(Negative is Better)', fontsize=14)
                ax4.set_xlabel('Correlation Coefficient', fontsize=12)
                ax4.axvline(x=0, color='black', linestyle='-')
                
                # Add coloring
                for i, bar in enumerate(ax4.patches):
                    if bar.get_width() < 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'performance_correlations.png', dpi=300, bbox_inches='tight')
            logger.info("Saved performance correlations plot")
        
        plt.close()
        
        all_correlations = []
        
        for col in df.columns:
            if col != 'position_numeric' and df[col].dtype in [np.float64, np.int64]:
                valid_df = df[position_filter & df[col].notna()]
                
                if len(valid_df) > 10:
                    corr = valid_df[['position_numeric', col]].corr().iloc[0, 1]
                    corr_significance = 'High' if abs(corr) > 0.3 else 'Medium' if abs(corr) > 0.15 else 'Low'
                    
                    all_correlations.append({
                        'feature': col,
                        'correlation_with_position': corr,
                        'significance': corr_significance,
                        'sample_size': len(valid_df)
                    })
        
        if all_correlations:
            corr_results = pd.DataFrame(all_correlations)
            corr_results = corr_results.sort_values('correlation_with_position')
            corr_results.to_csv(self.results_dir / 'feature_correlations.csv', index=False)
            logger.info("Saved correlation results to CSV")
    
    def analyze_player_consistency(self, min_tournaments: int = 10, save_plots: bool = True) -> None:
        logger.info("Analyzing player consistency")

        df = self.load_tournament_history(min_year=2019)
        if df.empty:
            logger.warning("No tournament data found for consistency analysis")
            return
    
        player_stats = df.groupby('player_id').agg({
            'player_name': 'first',
            'position_numeric': ['count', 'mean', 'std', 'min', 'max'],
            'par_relative': ['mean', 'std', 'min', 'max']
        })
        
        player_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in player_stats.columns]

        player_stats = player_stats[player_stats['position_numeric_count'] >= min_tournaments]
        
        if player_stats.empty:
            logger.warning(f"No players found with at least {min_tournaments} tournaments")
            return

        player_stats['position_cv'] = player_stats['position_numeric_std'] / player_stats['position_numeric_mean']
        player_stats['score_cv'] = player_stats['par_relative_std'] / abs(player_stats['par_relative_mean'])

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        
        scatter = ax1.scatter(
            player_stats['position_numeric_mean'],
            player_stats['position_cv'],
            alpha=0.7,
            s=player_stats['position_numeric_count']*2,
            c=player_stats['position_numeric_count'],
            cmap='viridis'
        )
        
        ax1.set_xlabel('Average Finish Position (lower is better)', fontsize=12)
        ax1.set_ylabel('Consistency (lower is more consistent)', fontsize=12)
        ax1.set_title('Player Performance vs. Consistency', fontsize=14)

        plt.colorbar(scatter, ax=ax1, label='Tournament Count')

        top_players = player_stats.nsmallest(5, 'position_numeric_mean')
        
        for idx, row in top_players.iterrows():
            ax1.annotate(
                row['player_name_first'],
                xy=(row['position_numeric_mean'], row['position_cv']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

        ax2 = fig.add_subplot(gs[0, 1])

        consistent_players = player_stats[player_stats['position_numeric_mean'] < 50]
        consistent_players = consistent_players.nsmallest(15, 'position_cv')
        
        sns.barplot(
            data=consistent_players.reset_index(),
            y='player_name_first',
            x='position_cv',
            ax=ax2
        )
        
        ax2.set_title('Most Consistent Players\n(Lower CV = More Consistent)', fontsize=14)
        ax2.set_xlabel('Consistency (CV of Position)', fontsize=12)
        ax2.set_ylabel('Player', fontsize=12)

        ax3 = fig.add_subplot(gs[1, 0])

        top_players = player_stats.nsmallest(10, 'position_numeric_mean')
        top_player_ids = top_players.index.tolist()

        top_player_data = df[df['player_id'].isin(top_player_ids)]

        sns.boxplot(
            data=top_player_data,
            x='player_name',
            y='par_relative',
            ax=ax3
        )
        
        ax3.set_title('Score Distribution of Top 10 Players', fontsize=14)
        ax3.set_xlabel('Player', fontsize=12)
        ax3.set_ylabel('Score to Par', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        ax4 = fig.add_subplot(gs[1, 1])

        round_scores = []
        
        for _, row in df.iterrows():
            player_id = row['player_id']
            player_name = row['player_name']
            
            for i in range(4): 
                round_col = f'round{i+1}_par_relative'
                
                if round_col in row and pd.notna(row[round_col]):
                    round_scores.append({
                        'player_id': player_id,
                        'player_name': player_name,
                        'round': i+1,
                        'score': row[round_col]
                    })
        
        if round_scores:
            round_df = pd.DataFrame(round_scores)

            top_round_avg = round_df[round_df['player_id'].isin(top_player_ids)]
            top_round_avg = top_round_avg.groupby(['player_name', 'round'])['score'].mean().reset_index()
            
            sns.lineplot(
                data=top_round_avg,
                x='round',
                y='score',
                hue='player_name',
                marker='o',
                ax=ax4
            )
            
            ax4.set_title('Round-by-Round Average Scores (Top Players)', fontsize=14)
            ax4.set_xlabel('Round', fontsize=12)
            ax4.set_ylabel('Average Score to Par', fontsize=12)
            ax4.legend(title='Player', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'player_consistency_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Saved player consistency analysis plot")
        
        plt.close()
        
        player_stats.to_csv(self.results_dir / 'player_consistency_stats.csv')
        logger.info("Saved player consistency data to CSV")
    
    def analyze_course_effects(self, save_plots: bool = True) -> None:
        logger.info("Analyzing course effects")

        course_stats = list(self.db.course_stats.find({}))
        
        if not course_stats:
            logger.warning("No course stats found for analysis")
            return

        tournament_results = self.load_tournament_history(min_year=2019)

        course_data = []
        hole_data = []
        
        for course in course_stats:
            course_id = course.get('course_id')
            tournament_id = course.get('tournament_id')
            course_name = course.get('course_name')
            par = course.get('par')
            yardage = course.get('yardage')
            
            if par is not None and yardage is not None:
                try:
                    par = float(par)
                    yardage = float(yardage)

                    course_data.append({
                        'course_id': course_id,
                        'tournament_id': tournament_id,
                        'course_name': course_name,
                        'par': par,
                        'yardage': yardage
                    })
                except (ValueError, TypeError):
                    continue

            for round_data in course.get('rounds', []):
                round_num = round_data.get('round_number')
                
                for hole in round_data.get('holes', []):
                    try:
                        hole_dict = {
                            'course_id': course_id,
                            'tournament_id': tournament_id,
                            'round': round_num,
                            'hole': hole.get('hole_number'),
                            'par': float(hole.get('par')) if hole.get('par') is not None else None,
                            'yards': float(hole.get('yards')) if hole.get('yards') is not None else None
                        }

                        hole_dict['scoringaveragedifference'] = self._safe_convert_to_numeric(hole.get('scoring_average_diff'))

                        hole_dict['eagles'] = self._safe_convert_to_numeric(hole.get('eagles'))
                        hole_dict['birdies'] = self._safe_convert_to_numeric(hole.get('birdies'))
                        hole_dict['pars'] = self._safe_convert_to_numeric(hole.get('pars'))
                        hole_dict['bogeys'] = self._safe_convert_to_numeric(hole.get('bogeys'))
                        hole_dict['doublebogeys'] = self._safe_convert_to_numeric(hole.get('double_bogeys'))
                        
                        hole_data.append(hole_dict)
                    except (ValueError, TypeError) as e:
                        continue
    
    def analyze_weather_impact(self, save_plots: bool = True) -> None:
        logger.info("Analyzing weather impact")
        

        weather_df = self.load_tournament_weather()
        
        if weather_df.empty:
            logger.warning("No weather data found for analysis")
            return
        
        tournament_results = self.load_tournament_history(min_year=2019)
        
        if tournament_results.empty:
            logger.warning("No tournament results found for weather analysis")
            return
        round_scores = []
        
        for tournament_id in tournament_results['tournament_id'].unique():
            tournament_df = tournament_results[tournament_results['tournament_id'] == tournament_id]
            round_cols = [col for col in tournament_df.columns if col.startswith('round') and col.endswith('_par_relative')]
            
            if round_cols:
                for i, col in enumerate(round_cols):
                    try:
                        valid_scores = tournament_df[col].dropna()
                        if len(valid_scores) > 0:
                            round_avg = valid_scores.mean()
                            
                            round_scores.append({
                                'tournament_id': tournament_id,
                                'round': i+1,
                                'avg_score': round_avg
                            })
                    except Exception as e:
                        logger.warning(f"Error calculating round average for {tournament_id}, round {i+1}: {str(e)}")
        
        if not round_scores:
            logger.warning("No round scores found for weather analysis")
            return
            
        round_scores_df = pd.DataFrame(round_scores)
        weather_round_data = []
        
        for _, row in weather_df.iterrows():
            tournament_id = row['tournament_id']
            
            for i in range(4): 
                round_temp_col = f'round{i+1}_temp'
                round_wind_col = f'round{i+1}_windspeed'
                round_precip_col = f'round{i+1}_precip'

                if round_temp_col in row and pd.notna(row[round_temp_col]):
                    weather_round = {
                        'tournament_id': tournament_id,
                        'round': i+1,
                        'temperature': row[round_temp_col]
                    }
                    
                    if round_wind_col in row and pd.notna(row[round_wind_col]):
                        weather_round['windspeed'] = row[round_wind_col]

                    if round_precip_col in row and pd.notna(row[round_precip_col]):
                        weather_round['precipitation'] = row[round_precip_col]
                    
                    weather_round_data.append(weather_round)
        weather_round_df = pd.DataFrame(weather_round_data)
        

        if weather_round_df.empty:
            logger.warning("No weather round data available for analysis")
            return
        merged_df = pd.merge(
            round_scores_df,
            weather_round_df,
            on=['tournament_id', 'round'],
            how='inner'
        )
        if merged_df.empty:
            logger.warning("No matching data found for weather analysis")
            return
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        temp_data = merged_df.dropna(subset=['temperature', 'avg_score'])
        
        if len(temp_data) > 5: 
            ax1.scatter(
                temp_data['temperature'],
                temp_data['avg_score'],
                alpha=0.7,
                s=30
            )

            try:
                if len(temp_data) > 1 and temp_data['temperature'].nunique() > 1:
                    z = np.polyfit(temp_data['temperature'], temp_data['avg_score'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(temp_data['temperature'].min(), temp_data['temperature'].max(), 100)
                    ax1.plot(x_range, p(x_range), "r--")

                    corr = temp_data[['temperature', 'avg_score']].corr().iloc[0, 1]
                    ax1.annotate(
                        f'Correlation: {corr:.3f}',
                        xy=(0.05, 0.95),
                        xycoords='axes fraction',
                        fontsize=12
                    )
            except np.linalg.LinAlgError:
                logger.warning("Could not compute trend line for temperature vs. scoring")
            except Exception as e:
                logger.warning(f"Error in temperature analysis: {str(e)}")
        
            ax1.set_title('Temperature vs. Scoring Average', fontsize=14)
            ax1.set_xlabel('Temperature (°F)', fontsize=12)
            ax1.set_ylabel('Average Score to Par', fontsize=12)
        else:
            ax1.text(0.5, 0.5, "Insufficient temperature data for analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)

        ax2 = fig.add_subplot(gs[0, 1])

        wind_data = merged_df.dropna(subset=['windspeed', 'avg_score'])
        
        if len(wind_data) >= 5: 
            ax2.scatter(
                wind_data['windspeed'],
                wind_data['avg_score'],
                alpha=0.7,
                s=30
            )

            try:
                if len(wind_data) > 1 and wind_data['windspeed'].nunique() > 1:
                    z = np.polyfit(wind_data['windspeed'], wind_data['avg_score'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(wind_data['windspeed'].min(), wind_data['windspeed'].max(), 100)
                    ax2.plot(x_range, p(x_range), "r--")
                    corr = wind_data[['windspeed', 'avg_score']].corr().iloc[0, 1]
                    ax2.annotate(
                        f'Correlation: {corr:.3f}',
                        xy=(0.05, 0.95),
                        xycoords='axes fraction',
                        fontsize=12
                    )
            except np.linalg.LinAlgError:
                logger.warning("Could not compute trend line for wind speed vs. scoring")
            except Exception as e:
                logger.warning(f"Error in wind speed analysis: {str(e)}")
            
            ax2.set_title('Wind Speed vs. Scoring Average', fontsize=14)
            ax2.set_xlabel('Wind Speed (mph)', fontsize=12)
            ax2.set_ylabel('Average Score to Par', fontsize=12)
        else:
            ax2.text(0.5, 0.5, "Insufficient wind data for analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)

        ax3 = fig.add_subplot(gs[1, 0])

        precip_data = merged_df.dropna(subset=['precipitation', 'avg_score'])
        
        if len(precip_data) >= 5:
            ax3.scatter(
                precip_data['precipitation'],
                precip_data['avg_score'],
                alpha=0.7,
                s=30
            )

            try:
                if len(precip_data) > 2 and precip_data['precipitation'].nunique() > 1:  
                    z = np.polyfit(precip_data['precipitation'], precip_data['avg_score'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(precip_data['precipitation'].min(), precip_data['precipitation'].max(), 100)
                    ax3.plot(x_range, p(x_range), "r--")

                    corr = precip_data[['precipitation', 'avg_score']].corr().iloc[0, 1]
                    ax3.annotate(
                        f'Correlation: {corr:.3f}',
                        xy=(0.05, 0.95),
                        xycoords='axes fraction',
                        fontsize=12
                    )
            except np.linalg.LinAlgError:
                logger.warning("Could not compute trend line for precipitation vs. scoring")
            except Exception as e:
                logger.warning(f"Error in precipitation analysis: {str(e)}")
            
            ax3.set_title('Precipitation vs. Scoring Average', fontsize=14)
            ax3.set_xlabel('Precipitation (inches)', fontsize=12)
            ax3.set_ylabel('Average Score to Par', fontsize=12)
        else:
            ax3.text(0.5, 0.5, "Insufficient precipitation data for analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
        
        ax4 = fig.add_subplot(gs[1, 1])
        
        try:
            if len(merged_df) >= 10:

                merged_df = merged_df.copy()
                merged_df['condition'] = 'Normal'

                if 'windspeed' in merged_df.columns:
                    wind_valid = merged_df.dropna(subset=['windspeed'])
                    if len(wind_valid) >= 5:
                        wind_threshold = wind_valid['windspeed'].quantile(0.75)
                        merged_df.loc[merged_df['windspeed'] > wind_threshold, 'condition'] = 'Windy'
       
                if 'precipitation' in merged_df.columns:
                    rain_valid = merged_df.dropna(subset=['precipitation'])
                    if len(rain_valid) >= 5:
                        merged_df.loc[merged_df['precipitation'] > 0.1, 'condition'] = 'Rain'
                
                if 'temperature' in merged_df.columns:
                    temp_valid = merged_df.dropna(subset=['temperature'])
                    if len(temp_valid) >= 5:
                        cold_threshold = temp_valid['temperature'].quantile(0.25)
                        merged_df.loc[merged_df['temperature'] < cold_threshold, 'condition'] = 'Cold'
    
                if 'temperature' in merged_df.columns:
                    temp_valid = merged_df.dropna(subset=['temperature'])
                    if len(temp_valid) >= 5:
                        hot_threshold = temp_valid['temperature'].quantile(0.75)
                        merged_df.loc[merged_df['temperature'] > hot_threshold, 'condition'] = 'Hot'
            
                condition_counts = merged_df['condition'].value_counts()
                valid_conditions = [cond for cond in condition_counts.index if condition_counts[cond] >= 3]
                
                if valid_conditions:
                    condition_data = merged_df[merged_df['condition'].isin(valid_conditions)]
                    
                    if not condition_data.empty:
                        sns.boxplot(
                            data=condition_data,
                            x='condition',
                            y='avg_score',
                            ax=ax4
                        )
                        
                        ax4.set_title('Scoring by Weather Condition', fontsize=14)
                        ax4.set_xlabel('Weather Condition', fontsize=12)
                        ax4.set_ylabel('Average Score to Par', fontsize=12)
                    else:
                        ax4.text(0.5, 0.5, "Insufficient data for condition analysis", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax4.transAxes, fontsize=12)
                else:
                    ax4.text(0.5, 0.5, "No weather conditions with sufficient data", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes, fontsize=12)
            else:
                ax4.text(0.5, 0.5, "Insufficient data for condition analysis", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
        except Exception as e:
            logger.warning(f"Error in weather condition analysis: {str(e)}")
            ax4.text(0.5, 0.5, "Error in weather condition analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'weather_impact_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Saved weather impact analysis plot")
        
        plt.close()
    
    def analyze_feature_importance(self, save_plots: bool = True) -> None:
        logger.info("Analyzing feature importance")
        
        df = self.create_consolidated_dataset()
        
        if df.empty:
            logger.warning("No data found for feature importance analysis")
            return
        if 'position_numeric' not in df.columns:
            df['position_numeric'] = df['position'].apply(self._parse_position)
        df = df[df['position_numeric'].notna()]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        cols_to_remove = ['position_numeric', '_id']
        cols_to_remove.extend([col for col in numeric_cols if 'collected_at' in col])
        
        feature_cols = [col for col in numeric_cols if col not in cols_to_remove]
        valid_features = []
        for col in feature_cols:
            missing_pct = df[col].isna().mean()
            if missing_pct < 0.5:  
                valid_features.append(col)
        
        if not valid_features:
            logger.warning("No valid features found for importance analysis")
            return

        X = df[valid_features].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        y = df['position_numeric']

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            logger.warning(f"Error standardizing features: {str(e)}")
            X_scaled = np.zeros(X.shape)
            for i, col in enumerate(X.columns):
                mean = X[col].mean()
                std = X[col].std()
                if std > 0:
                    X_scaled[:, i] = (X[col] - mean) / std
                else:
                    X_scaled[:, i] = 0

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])

        correlations = []
        for i, col in enumerate(valid_features):
            try:

                valid_mask = ~np.isnan(X[col]) & ~np.isnan(y)
                if valid_mask.sum() > 5: 
                    corr = np.corrcoef(X[col][valid_mask], y[valid_mask])[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))
            except Exception as e:
                logger.warning(f"Error calculating correlation for {col}: {str(e)}")
                continue
        
        if not correlations:
            ax1.text(0.5, 0.5, "No valid correlations found", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)
        else:
            correlations.sort(key=lambda x: x[1], reverse=True)
        
            top_correlations = correlations[:min(15, len(correlations))]

            corr_df = pd.DataFrame(top_correlations, columns=['feature', 'abs_correlation'])
            
            try:
                sns.barplot(
                    data=corr_df,
                    y='feature',
                    x='abs_correlation',
                    ax=ax1
                )
                
                ax1.set_title('Top Features by Correlation with Position', fontsize=14)
                ax1.set_xlabel('Absolute Correlation', fontsize=12)
                ax1.set_ylabel('Feature', fontsize=12)
            except Exception as e:
                logger.warning(f"Error creating correlation plot: {str(e)}")
                ax1.text(0.5, 0.5, "Error creating correlation plot", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=12)
    
        ax2 = fig.add_subplot(gs[0, 1])
        
        try:

            variances = []
            for i, col in enumerate(valid_features):
                var = np.var(X_scaled[:, i])
                if not np.isnan(var):
                    variances.append((col, var))
            
            if not variances:
                ax2.text(0.5, 0.5, "No valid variance data", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
            else:
                variances.sort(key=lambda x: x[1], reverse=True)
                
                top_variances = variances[:min(15, len(variances))]
                
                var_df = pd.DataFrame(top_variances, columns=['feature', 'variance'])
                
                sns.barplot(
                    data=var_df,
                    y='feature',
                    x='variance',
                    ax=ax2
                )
                
                ax2.set_title('Top Features by Variance', fontsize=14)
                ax2.set_xlabel('Variance', fontsize=12)
                ax2.set_ylabel('Feature', fontsize=12)
        except Exception as e:
            logger.warning(f"Error analyzing feature variances: {str(e)}")
            ax2.text(0.5, 0.5, "Error analyzing feature variances", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
        
        ax3 = fig.add_subplot(gs[1, 0])
        
        try:
            if X_scaled.shape[0] > 10 and X_scaled.shape[1] > 1:

                n_components = min(10, len(valid_features))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                pca.fit(X_scaled)
                explained_var = pca.explained_variance_ratio_
                cum_explained_var = np.cumsum(explained_var)
                
                ax3.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
                ax3.step(range(1, len(cum_explained_var) + 1), cum_explained_var, where='mid', 
                        label='Cumulative Explained Variance')
                
                ax3.set_title('PCA Explained Variance', fontsize=14)
                ax3.set_xlabel('Principal Component', fontsize=12)
                ax3.set_ylabel('Explained Variance Ratio', fontsize=12)
                ax3.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, "Insufficient data for PCA", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=12)
        except Exception as e:
            logger.warning(f"Error in PCA analysis: {str(e)}")
            ax3.text(0.5, 0.5, "Error in PCA analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)

        ax4 = fig.add_subplot(gs[1, 1])
        
        try:
            feature_groups = {
                'Strokes Gained': [col for col in valid_features if 'sg_' in col.lower()],
                'Course Fit': [col for col in valid_features if 'course_fit' in col.lower()],
                'Recent Form': [col for col in valid_features if 'recent_' in col.lower()],
                'Historical': [col for col in valid_features if 'historical_' in col.lower() or 'hist_' in col.lower()],
                'Weather': [col for col in valid_features if any(w in col.lower() for w in ['temp', 'wind', 'precip', 'humidity'])]
            }
            
            group_correlations = []
            
            for group_name, group_cols in feature_groups.items():
                if not group_cols:
                    continue

                valid_group_cols = [col for col in group_cols if col in X.columns]
                
                if valid_group_cols:
                    group_corrs = []
                    for col in valid_group_cols:
                        try:

                            valid_mask = ~np.isnan(X[col]) & ~np.isnan(y)
                            if valid_mask.sum() > 5:  
                                corr = abs(np.corrcoef(X[col][valid_mask], y[valid_mask])[0, 1])
                                if not np.isnan(corr):
                                    group_corrs.append(corr)
                        except Exception:
                            continue
                    
                    if group_corrs:
                        avg_corr = np.mean(group_corrs)
                        group_correlations.append((group_name, avg_corr, len(valid_group_cols)))
            

            if not group_correlations:
                ax4.text(0.5, 0.5, "No valid feature groups found", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
            else:
                group_corr_df = pd.DataFrame(group_correlations, columns=['group', 'avg_correlation', 'feature_count'])
                group_corr_df = group_corr_df.sort_values('avg_correlation', ascending=False)
                
                bars = sns.barplot(
                    data=group_corr_df,
                    y='group',
                    x='avg_correlation',
                    ax=ax4
                )
                
    
                for i, row in enumerate(group_corr_df.itertuples()):
                    bars.text(row.avg_correlation + 0.01, i, f'n={row.feature_count}', va='center')
                
                ax4.set_title('Average Feature Importance by Group', fontsize=14)
                ax4.set_xlabel('Average Absolute Correlation', fontsize=12)
                ax4.set_ylabel('Feature Group', fontsize=12)
        except Exception as e:
            logger.warning(f"Error in feature group analysis: {str(e)}")
            ax4.text(0.5, 0.5, "Error in feature group analysis", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Saved feature importance analysis plot")
        
        plt.close()
    
        if correlations:
            try:
                importance_df = pd.DataFrame(correlations, columns=['feature', 'abs_correlation'])
                importance_df = importance_df.sort_values('abs_correlation', ascending=False)
                importance_df.to_csv(self.results_dir / 'feature_importance.csv', index=False)
                logger.info("Saved feature importance data to CSV")
            except Exception as e:
                logger.warning(f"Error saving feature importance data: {str(e)}")
                
    def run_comprehensive_analysis(self) -> None:
        """Run all analysis functions to generate a complete set of insights"""
        logger.info("Starting comprehensive golf data analysis")
        
        self.results_dir.mkdir(exist_ok=True)
    
        try:
            self.analyze_tournament_outcomes()
        except Exception as e:
            logger.error(f"Error in tournament outcomes analysis: {str(e)}")
        
        try:
            self.analyze_performance_correlations()
        except Exception as e:
            logger.error(f"Error in performance correlations analysis: {str(e)}")
        
        try:
            self.analyze_player_consistency()
        except Exception as e:
            logger.error(f"Error in player consistency analysis: {str(e)}")
        
        try:
            self.analyze_course_effects()
        except Exception as e:
            logger.error(f"Error in course effects analysis: {str(e)}")
        
        try:
            self.analyze_weather_impact()
        except Exception as e:
            logger.error(f"Error in weather impact analysis: {str(e)}")
        
        try:
            self.analyze_feature_importance()
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
        
        try:
            self._generate_summary_report()
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
        
        try:
            self._generate_summary_report()
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")

        logger.info("Comprehensive analysis complete. Results saved to: " + str(self.results_dir))
    
    def run_comprehensive_analysis(self) -> None:
        logger.info("Starting comprehensive golf data analysis")
    
    
        self.results_dir.mkdir(exist_ok=True)
        
        self.analyze_tournament_outcomes()
        self.analyze_performance_correlations()
        self.analyze_player_consistency()
        self.analyze_course_effects()
        self.analyze_weather_impact()
        self.analyze_feature_importance()
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("Comprehensive analysis complete. Results saved to: " + str(self.results_dir))
    
    def _generate_summary_report(self) -> None:
        """Generate a summary report of key findings"""
        try:
            # Check which result files exist
            feature_importance_path = self.results_dir / 'feature_importance.csv'
            player_consistency_path = self.results_dir / 'player_consistency_stats.csv'
            feature_correlations_path = self.results_dir / 'feature_correlations.csv'
            
            report_parts = []
            report_parts.append("# Golf Tournament Prediction Analysis Summary")
            report_parts.append(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_parts.append("\n## Key Findings\n")
            
            # Add feature importance summary if available
            if feature_importance_path.exists():
                try:
                    importance_df = pd.read_csv(feature_importance_path)
                    importance_df = importance_df.sort_values('abs_correlation', ascending=False)
                    
                    report_parts.append("### Top Predictive Features (by correlation with position)")
                    
                    # Add top 10 features
                    top_features = importance_df.head(10)
                    feature_rows = []
                    for _, row in top_features.iterrows():
                        feature_rows.append(f"- **{row['feature']}**: {row['abs_correlation']:.3f}")
                    
                    report_parts.append("\n".join(feature_rows))
                except Exception as e:
                    logger.warning(f"Error reading feature importance data: {str(e)}")
                    report_parts.append("### Feature Importance\nUnable to process feature importance data.")
            
            # Add player consistency summary if available
            if player_consistency_path.exists():
                try:
                    consistency_df = pd.read_csv(player_consistency_path)
                    
                    # Filter out potentially problematic entries
                    consistency_df = consistency_df[consistency_df['position_numeric_mean'].notna()]
                    
                    if not consistency_df.empty:
                        # Top performers
                        consistency_df = consistency_df.sort_values('position_numeric_mean')
                        top_players = consistency_df.head(5)
                        
                        report_parts.append("\n### Top Performing Players")
                        
                        player_rows = []
                        for _, row in top_players.iterrows():
                            player_name = row.get('player_name_first', 'Unknown')
                            avg_pos = row.get('position_numeric_mean', 0)
                            cv = row.get('position_cv', 0)
                            
                            player_rows.append(f"- **{player_name}**: Avg. Position {avg_pos:.1f}, " +
                                            f"Consistency (CV) {cv:.2f}")
                        
                        report_parts.append("\n".join(player_rows))
                        
                        # Most consistent players
                        valid_consistency = consistency_df[consistency_df['position_numeric_mean'] < 50]  # Focus on decent performers
                        
                        if not valid_consistency.empty:
                            valid_consistency = valid_consistency.sort_values('position_cv')
                            consistent_players = valid_consistency.head(5)
                            
                            report_parts.append("\n### Most Consistent Players")
                            
                            player_rows = []
                            for _, row in consistent_players.iterrows():
                                player_name = row.get('player_name_first', 'Unknown')
                                avg_pos = row.get('position_numeric_mean', 0)
                                cv = row.get('position_cv', 0)
                                
                                player_rows.append(f"- **{player_name}**: Consistency (CV) {cv:.2f}, " +
                                                f"Avg. Position {avg_pos:.1f}")
                            
                            report_parts.append("\n".join(player_rows))
                except Exception as e:
                    logger.warning(f"Error reading player consistency data: {str(e)}")
                    report_parts.append("### Player Consistency\nUnable to process player consistency data.")
            
            # Add correlation analysis if available
            if feature_correlations_path.exists():
                try:
                    corr_df = pd.read_csv(feature_correlations_path)
                    
                    # Get high significance correlations
                    high_sig = corr_df[corr_df['significance'] == 'High']
                    
                    if not high_sig.empty:
                        high_sig = high_sig.sort_values('correlation_with_position')
                        
                        report_parts.append("\n### Strongest Predictive Relationships")
                        
                        pos_corr = high_sig[high_sig['correlation_with_position'] > 0].head(5)
                        if not pos_corr.empty:
                            report_parts.append("\nFactors associated with **worse performance** (higher position number):")
                            
                            corr_rows = []
                            for _, row in pos_corr.iterrows():
                                corr_rows.append(f"- **{row['feature']}**: {row['correlation_with_position']:.3f}")
                            
                            report_parts.append("\n".join(corr_rows))
                        
                       
                        neg_corr = high_sig[high_sig['correlation_with_position'] < 0].head(5)
                        if not neg_corr.empty:
                            report_parts.append("\nFactors associated with **better performance** (lower position number):")
                            
                            corr_rows = []
                            for _, row in neg_corr.iterrows():
                                corr_rows.append(f"- **{row['feature']}**: {row['correlation_with_position']:.3f}")
                            
                            report_parts.append("\n".join(corr_rows))
                except Exception as e:
                    logger.warning(f"Error reading correlation data: {str(e)}")
                    report_parts.append("### Feature Correlations\nUnable to process correlation data.")
            
            
            
            report_text = "\n\n".join(report_parts)
            with open(self.results_dir / 'analysis_summary.md', 'w') as f:
                f.write(report_text)
                
            logger.info("Generated summary report")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")


def main():
    try:
        load_dotenv()
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            print("Error: MongoDB URI not found in environment variables")
            return
        analyzer = GolfDataAnalyzer(uri=mongodb_uri)
        analyzer.run_comprehensive_analysis()
        print("Analysis complete! Results saved to: analysis_results/")
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        traceback.print_exc()  


if __name__ == "__main__":
    main()