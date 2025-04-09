# data_extractor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataExtractor initialized")
    
    def list_collections(self):
        try:
            collections = self.db_manager.client[self.db_manager.db.name].list_collection_names()
            self.logger.info(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            self.logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    def get_collection_info(self, collection_name):
        try:
            return self.db_manager.get_collection_stats(collection_name)
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}
    
    def extract_player_stats(self, seasons=None, player_ids=None, stat_categories=None):
        self.logger.info("Extracting player statistics")
        query = {}
        if seasons:
            if isinstance(seasons, int):
                query["season"] = seasons
            else:
                query["season"] = {"$in": seasons}
        
        if player_ids:
            query["player_id"] = {"$in": player_ids}
        
        try:
            player_stats_raw = self.db_manager.run_query("player_stats", query)
            
            if not player_stats_raw:
                self.logger.warning("No player stats found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found stats for {len(player_stats_raw)} player-seasons")
        
            flattened_data = self._flatten_player_stats(
                player_stats_raw, 
                categories=stat_categories
            )

            df = pd.DataFrame(flattened_data)

            df = self._convert_player_stats_types(df)
            
            self.logger.info(f"Extracted player stats with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract player stats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def extract_collection_to_df(self, collection_name, query=None,  projection=None):
        try:
            if query is None:
                query = {}
                
            results = self.db_manager.run_query(collection_name, query, projection)
        
            if not results:
                self.logger.warning(f"No data found in {collection_name} for the given query")
                return pd.DataFrame()

            df = pd.DataFrame(results)
            
            self.logger.info(f"Extracted {len(df)} records from {collection_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract {collection_name}: {str(e)}")
            return pd.DataFrame()
    
    def _flatten_player_stats(self, 
                             player_stats_raw: List[Dict], 
                             categories: Optional[List[str]] = None) -> List[Dict]:
        flattened_data = []
        
        for player_doc in player_stats_raw:
            player_base = {
                "player_id": player_doc.get("player_id"),
                "name": player_doc.get("name"),
                "season": player_doc.get("season"),
                "collected_at": player_doc.get("collected_at")
            }
        
            stats = player_doc.get("stats", {})
            if categories:
                stats = {k: v for k, v in stats.items() if k in categories}

            flattened_stats = {}
            for category, stat_list in stats.items():

                clean_category = category.replace(", ", "_").lower()
                
                for stat in stat_list:
                    stat_title = stat.get("title", "")
                    stat_id = stat.get("stat_id", "")
                    
                    clean_title = stat_title.lower().replace(" ", "_").replace(":", "").replace("-", "_")
                    clean_title = clean_title.replace("(", "").replace(")", "").replace("%", "pct")
                    
                    # For uniqueness, combine category and title
                    col_name = f"{clean_category}_{clean_title}"
                    
                    # Extract the value
                    flattened_stats[col_name] = stat.get("value")
                    
                    # Add rank as a separate column
                    flattened_stats[f"{col_name}_rank"] = stat.get("rank")
                    
                    # Add supporting stats if they exist
                    if stat.get("supporting_stat_description") and stat.get("supporting_stat_value"):
                        supp_desc = stat.get("supporting_stat_description").lower().replace(" ", "_")
                        flattened_stats[f"{col_name}_{supp_desc}"] = stat.get("supporting_stat_value")
            
            # Combine base info with flattened stats
            player_entry = {**player_base, **flattened_stats}
            flattened_data.append(player_entry)
        
        return flattened_data
    
    def _convert_player_stats_types(self, df):
        df_converted = df.copy()
        
        for col in df_converted.columns:
            if col in ['player_id', 'name', 'collected_at']:
                continue
            if col == 'season':
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
                continue
            try:
                if df_converted[col].astype(str).str.contains('%').any():

                    df_converted[col] = df_converted[col].astype(str).str.replace('%', '').astype(float) / 100

                elif df_converted[col].astype(str).str.contains('\\$').any():

                    df_converted[col] = df_converted[col].astype(str).str.replace('[\\$,]', '', regex=True).astype(float)

                elif df_converted[col].astype(str).str.contains("'").any():
                    def feet_inches_to_inches(value):
                        try:
                            if isinstance(value, (int, float)):
                                return value
                            if pd.isna(value) or value == '':
                                return np.nan
                            parts = value.split("'")
                            feet = float(parts[0])
                            inches = float(parts[1]) if len(parts) > 1 and parts[1].strip() else 0
                            return feet * 12 + inches
                        except:
                            return np.nan
                    df_converted[col] = df_converted[col].apply(feet_inches_to_inches)
                else:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            except Exception as e:
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted
    
    def sample_collection(self, collection_name, n):
        try:
            pipeline = [{"$sample": {"size": n}}]
            samples = list(self.db_manager.db[collection_name].aggregate(pipeline))
            
            return samples
        except Exception as e:
            self.logger.error(f"Failed to sample {collection_name}: {str(e)}")
            return []
        
        
        ## TOURNAMENT HISTORY 
        # Add these methods to your DataExtractor class

    def extract_tournament_history(self,
                                tournament_ids: Optional[Union[str, List[str]]] = None,
                                years: Optional[Union[int, List[int]]] = None,
                                player_ids: Optional[List[str]] = None) -> pd.DataFrame:
        
        self.logger.info("Extracting tournament history")

        query = {}
        
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["tournament_id"] = tournament_ids
            else:
                query["tournament_id"] = {"$in": tournament_ids}
        
        if years:
            if isinstance(years, int):
                query["year"] = years
            else:
                query["year"] = {"$in": years}
        
        try:
            history_raw = self.db_manager.run_query("tournament_history", query)
            
            if not history_raw:
                self.logger.warning("No tournament history found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(history_raw)} tournament history records")
            
            if player_ids:
                self.logger.info(f"Filtering results for {len(player_ids)} players")
                flattened_data = self._flatten_tournament_history_by_players(history_raw, player_ids)
            else:
                flattened_data = self._flatten_tournament_history(history_raw)

            df = pd.DataFrame(flattened_data)

            df = self._convert_tournament_history_types(df)
            
            self.logger.info(f"Extracted tournament history with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract tournament history: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _flatten_tournament_history(self, history_raw):
        flattened_data = []
        for tournament_doc in history_raw:
            tournament_base = {
                "tournament_id": tournament_doc.get("tournament_id"),
                "original_tournament_id": tournament_doc.get("original_tournament_id"),
                "year": tournament_doc.get("year"),
                "collected_at": tournament_doc.get("collected_at")
            }
            
            players = tournament_doc.get("players", [])
            tournament_base["player_count"] = len(players)
            if players and len(players) > 0:
                sorted_players = sorted(players, key=lambda p: p.get("position", "999"))
                if sorted_players[0].get("position") == "1":
                    winner = sorted_players[0]
                    tournament_base["winner_id"] = winner.get("player_id")
                    tournament_base["winner_name"] = winner.get("name")
                    tournament_base["winning_score"] = winner.get("total_score")
                    tournament_base["winning_score_to_par"] = winner.get("par_relative")
            
            flattened_data.append(tournament_base)
        
        return flattened_data

    def _flatten_tournament_history_by_players(self, history_raw, player_ids):
        flattened_data = []
        
        for tournament_doc in history_raw:

            tournament_base = {
                "tournament_id": tournament_doc.get("tournament_id"),
                "original_tournament_id": tournament_doc.get("original_tournament_id"),
                "year": tournament_doc.get("year"),
                "collected_at": tournament_doc.get("collected_at")
            }

            all_players = tournament_doc.get("players", [])

            if player_ids:
                players = [p for p in all_players if p.get("player_id") in player_ids]
            else:
                players = all_players
            
            for player in players:
                player_entry = tournament_base.copy()
                
                player_entry["player_id"] = player.get("player_id")
                player_entry["player_name"] = player.get("name")
                player_entry["position"] = player.get("position")
                player_entry["country"] = player.get("country")
                player_entry["total_score"] = player.get("total_score")
                player_entry["score_to_par"] = player.get("par_relative")

                rounds = player.get("rounds", [])
                for i, round_data in enumerate(rounds):
                    round_num = i + 1
                    player_entry[f"round{round_num}_score"] = round_data.get("score")
                    player_entry[f"round{round_num}_to_par"] = round_data.get("par_relative")
                
                flattened_data.append(player_entry)
        
        return flattened_data

    def _convert_tournament_history_types(self, df):
        df_converted = df.copy()
        numeric_columns = [
            'year', 'player_count', 'winning_score', 'winning_score_to_par',
            'total_score', 'score_to_par'
        ]
        
        for i in range(1, 5):  
            round_cols = [f'round{i}_score', f'round{i}_to_par']
            numeric_columns.extend(round_cols)
        
        if 'position' in df_converted.columns:
            df_converted['position_numeric'] = df_converted['position'].apply(
                lambda x: pd.to_numeric(x.replace('T', ''), errors='coerce') if isinstance(x, str) else x
            )
        
        for col in [c for c in numeric_columns if c in df_converted.columns]:
            try:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except Exception as e:
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
        
        if 'year' in df_converted.columns:
            df_converted['year'] = df_converted['year'].astype('Int64')
        
        return df_converted

    def extract_tournament_performance_stats(self, player_id, min_years):
        self.logger.info(f"Extracting tournament performance stats for player {player_id}")
        
        try:
            player_history = self.extract_tournament_history(player_ids=[player_id])
            if player_history.empty:
                self.logger.warning(f"No tournament history found for player {player_id}")
                return pd.DataFrame()
            grouped = player_history.groupby('tournament_id')
            
            tournament_stats = []
            
            for tournament_id, group in grouped:
                if len(group) >= min_years:
                    stats = {
                        'tournament_id': tournament_id,
                        'player_id': player_id,
                        'appearances': len(group),
                        'avg_finish': group['position_numeric'].mean(),
                        'best_finish': group['position_numeric'].min(),
                        'worst_finish': group['position_numeric'].max(),
                        'avg_score_to_par': group['score_to_par'].mean(),
                        'cuts_made': sum(group['position_numeric'].notnull()),
                        'cuts_made_pct': sum(group['position_numeric'].notnull()) / len(group) * 100,
                        'top_10_finishes': sum(group['position_numeric'] <= 10),
                        'top_25_finishes': sum(group['position_numeric'] <= 25),
                        'years_played': group['year'].tolist(),
                        'last_appearance': group['year'].max(),
                        'last_finish': group.loc[group['year'].idxmax(), 'position_numeric'] if not group.empty else None
                    }
                    
                    tournament_stats.append(stats)
            
            return pd.DataFrame(tournament_stats)
            
        except Exception as e:
            self.logger.error(f"Failed to extract tournament performance stats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
        
        # WEATHER DATA EXTRACTION
    def extract_tournament_weather(self,tournament_ids=None,years=None):
        self.logger.info("Extracting tournament weather data")
        query = {}
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["tournament_id"] = tournament_ids
            else:
                query["tournament_id"] = {"$in": tournament_ids}
        if years:
            if isinstance(years, int):
                query["year"] = years
            else:
                query["year"] = {"$in": years}
        try:
            weather_raw = self.db_manager.run_query("tournament_weather", query)
            
            if not weather_raw:
                self.logger.warning("No tournament weather data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found weather data for {len(weather_raw)} tournaments")
            
            flattened_data = []
            
            for weather_doc in weather_raw:
                tournament_base = {
                    "tournament_id": weather_doc.get("tournament_id"),
                    "tournament_name": weather_doc.get("tournament_name"),
                    "course_name": weather_doc.get("course_name"),
                    "year": weather_doc.get("year"),
                    "location": weather_doc.get("location"),
                    "collected_at": weather_doc.get("collected_at")
                }
                rounds_data = weather_doc.get("rounds", [])
                if rounds_data:
                    for i, round_data in enumerate(rounds_data):
                        round_num = i + 1
                        
                        for weather_key, weather_value in round_data.items():
                            column_name = f"round{round_num}_{weather_key}"
                            tournament_base[column_name] = weather_value
                    
                    self._add_tournament_weather_averages(tournament_base, rounds_data)
                    flattened_data.append(tournament_base)
                else:
                    flattened_data.append(tournament_base)

            df = pd.DataFrame(flattened_data)
            df = self._convert_weather_types(df)
            self.logger.info(f"Extracted tournament weather with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract tournament weather: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def extract_tournament_weather_by_round(self, tournament_ids=None, years=None):
        self.logger.info("Extracting tournament weather data by round")
        query = {}
        
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["tournament_id"] = tournament_ids
            else:
                query["tournament_id"] = {"$in": tournament_ids}
        if years:
            if isinstance(years, int):
                query["year"] = years
            else:
                query["year"] = {"$in": years}
        try:
            weather_raw = self.db_manager.run_query("tournament_weather", query)
            
            if not weather_raw:
                self.logger.warning("No tournament weather data found for the given criteria")
                return pd.DataFrame()
    
            self.logger.info(f"Found weather data for {len(weather_raw)} tournaments")
            flattened_data = []
            for weather_doc in weather_raw:
                tournament_base = {
                    "tournament_id": weather_doc.get("tournament_id"),
                    "tournament_name": weather_doc.get("tournament_name"),
                    "course_name": weather_doc.get("course_name"),
                    "year": weather_doc.get("year"),
                    "location": weather_doc.get("location"),
                    "collected_at": weather_doc.get("collected_at")
                }
                rounds_data = weather_doc.get("rounds", [])
                
                for i, round_data in enumerate(rounds_data):
                    round_entry = tournament_base.copy()
                    round_entry["round_number"] = i + 1
                    
                    for weather_key, weather_value in round_data.items():
                        round_entry[weather_key] = weather_value
                    
                    flattened_data.append(round_entry)
            df = pd.DataFrame(flattened_data)
        
            df = self._convert_weather_types(df)
            
            self.logger.info(f"Extracted tournament weather by round with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract tournament weather by round: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _add_tournament_weather_averages(self, tournament_data: Dict, rounds_data: List[Dict]) -> None:
        key_metrics = ["temp", "tempmax", "tempmin", "humidity", "windspeed", "windgust", "cloudcover", "precip", "precipprob"]
        for metric in key_metrics:
            values = [round_data.get(metric) for round_data in rounds_data if metric in round_data]
            if values:
                tournament_data[f"avg_{metric}"] = sum(values) / len(values)
        
        conditions = [round_data.get("conditions") for round_data in rounds_data if "conditions" in round_data]
        if conditions:
            from collections import Counter
            conditions_counter = Counter(conditions)
            tournament_data["most_common_conditions"] = conditions_counter.most_common(1)[0][0]
            
        precip_values = [round_data.get("precip", 0) for round_data in rounds_data]
        if precip_values:
            tournament_data["total_precip"] = sum(precip_values)

    def _convert_weather_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_converted = df.copy()

        non_numeric_cols = [ 'tournament_id', 'tournament_name', 'course_name', 'location', 'collected_at', 'most_common_conditions' ]
    
        time_cols = ['collected_at']

        time_pattern_cols = [col for col in df_converted.columns if any(
            pattern in col for pattern in ['datetime', 'sunrise', 'sunset']
        )]
        time_cols.extend(time_pattern_cols)
        
        for col in time_cols:
            if col in df_converted.columns:
                try:
                    if 'sunrise' in col or 'sunset' in col:

                        pass 
                    elif 'datetime' in col:
                        df_converted[col] = pd.to_datetime(df_converted[col])
                    else:
                        df_converted[col] = pd.to_datetime(df_converted[col])
                except Exception as e:
                    self.logger.debug(f"Could not convert {col} to datetime: {str(e)}")
        

        if 'year' in df_converted.columns:
            df_converted['year'] = df_converted['year'].astype('Int64')
        
        if 'round_number' in df_converted.columns:
            df_converted['round_number'] = df_converted['round_number'].astype('Int64')
        
        # Convert numeric columns
        for col in df_converted.columns:
            if col not in non_numeric_cols and col not in time_cols:
                try:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                except Exception as e:
                    self.logger.debug(f"Could not convert column {col}: {str(e)}")
        
        return df_converted
    
    
    
    ## PLAYER CAREER




    def extract_player_career(self, player_ids = None,tour_code = "R"):
        self.logger.info("Extracting player career data")
        

        query = {}
        
        if player_ids:
            query["player_id"] = {"$in": player_ids}
        
        if tour_code:
            query["tour_code"] = tour_code
        
        try:
            career_raw = self.db_manager.run_query("player_career", query)
            
            if not career_raw:
                self.logger.warning("No player career data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found career data for {len(career_raw)} players")

            flattened_data = self._flatten_player_career(career_raw)

            df = pd.DataFrame(flattened_data)

            df = self._convert_player_career_types(df)
            
            self.logger.info(f"Extracted player career data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract player career data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def extract_player_career_yearly(self, player_ids = None,years = None,tour_code = "R"):
        self.logger.info("Extracting player career yearly data")
        query = {}
        
        if player_ids:
            query["player_id"] = {"$in": player_ids}
        
        if tour_code:
            query["tour_code"] = tour_code
        
        try:
            career_raw = self.db_manager.run_query("player_career", query)
            
            if not career_raw:
                self.logger.warning("No player career data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found career data for {len(career_raw)} players")
            flattened_yearly_data = []
            
            for career_doc in career_raw:
                player_id = career_doc.get("player_id")
                years_data = career_doc.get("years", [])

                if years:
                    if isinstance(years, int):
                        years_data = [y for y in years_data if y.get("year") == years]
                    else:
                        years_data = [y for y in years_data if y.get("year") in years]
                for year_data in years_data:
                    year_entry = {**year_data, "player_id": player_id}
                    flattened_yearly_data.append(year_entry)
            
            if not flattened_yearly_data:
                self.logger.warning("No yearly data found for the given criteria")
                return pd.DataFrame()

            df = pd.DataFrame(flattened_yearly_data)
            df = self._convert_player_yearly_types(df)
            
            self.logger.info(f"Extracted player yearly data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract player yearly data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _flatten_player_career(self, career_raw):
        flattened_data = []
        
        for career_doc in career_raw:
            player_data = {
                "player_id": career_doc.get("player_id"),
                "tour_code": career_doc.get("tour_code"),
                "collected_at": career_doc.get("collected_at"),
                "events": career_doc.get("events"),
                "wins": career_doc.get("wins"),
                "international_wins": career_doc.get("international_wins"),
                "major_wins": career_doc.get("major_wins"),
                "cuts_made": career_doc.get("cuts_made"),
                "runner_up": career_doc.get("runner_up"),
                "second": career_doc.get("second"),
                "third": career_doc.get("third"),
                "top10": career_doc.get("top10"),
                "top25": career_doc.get("top25"),
                "official_money": career_doc.get("official_money")
            }

            achievements = career_doc.get("achievements", [])
            for achievement in achievements:
                title = achievement.get("title", "").lower().replace(" ", "_")
                value = achievement.get("value", "")

                player_data[f"achievement_{title}"] = value
            
            tables = career_doc.get("tables", [])
            for table in tables:
                table_name = table.get("table_name", "").lower().replace(" ", "_")
                rows = table.get("rows", [])

                if rows:
                    row_contents = []
                    for row in rows:
                        row_title = row.get("row_title", "")
                        row_content = row.get("row_content", "")
                        if row_title and row_content:
                            row_contents.append(f"{row_title}: {row_content}")
                    
                    player_data[f"table_{table_name}"] = ", ".join(row_contents)
            

            years_data = career_doc.get("years", [])
            if years_data:
                active_years = [y.get("year") for y in years_data if y.get("year") is not None]
                
                if active_years:
                    player_data["first_year"] = min(active_years)
                    player_data["last_year"] = max(active_years)
                    player_data["career_span"] = player_data["last_year"] - player_data["first_year"] + 1
            
            flattened_data.append(player_data)
        
        return flattened_data

    def _convert_player_career_types(self, df):
        df_converted = df.copy()
        string_cols = ['player_id', 'tour_code', 'collected_at']
        money_cols = [col for col in df_converted.columns if 'money' in col.lower() or 'earnings' in col.lower()]

        for col in df_converted.columns:
            if col in string_cols:
                continue
            
            try:
                if col in money_cols:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].astype(str).str.replace('[\\$,]', '', regex=True)
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')

                elif df_converted[col].astype(str).str.contains('/').any():
                    fractions = df_converted[col].astype(str).str.split('/', expand=True)
                    if len(fractions.columns) == 2:
                        df_converted[f"{col}_numerator"] = pd.to_numeric(fractions[0], errors='coerce')
                        df_converted[f"{col}_denominator"] = pd.to_numeric(fractions[1], errors='coerce')

                        df_converted[f"{col}_pct"] = df_converted[f"{col}_numerator"] / df_converted[f"{col}_denominator"]

                else:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            except Exception as e:
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted

    def _convert_player_yearly_types(self, df):
        df_converted = df.copy()
        string_cols = ['player_id', 'tour_code', 'display_season']
        money_cols = [col for col in df_converted.columns if 'money' in col.lower() or 'earnings' in col.lower()]
        for col in df_converted.columns:
            if col in string_cols:
                continue
            
            try:
                if col in money_cols:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].replace('-', None)
                        df_converted[col] = df_converted[col].astype(str).str.replace('[\\$,]', '', regex=True)
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                else:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].replace('-', None)
                    
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            except Exception as e:
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted
    
    
    
    ### PLAYER CAREER OVERVIEW
    
    def extract_player_profile(self, player_ids = None):
        self.logger.info("Extracting player profile overview data")
        query = {}
        
        if player_ids:
            query["player_id"] = {"$in": player_ids}
        
        try:
            profile_raw = self.db_manager.run_query("player_profile_overview", query)
            
            if not profile_raw:
                self.logger.warning("No player profile data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found profile data for {len(profile_raw)} players")
            flattened_data = self._flatten_player_profile(profile_raw)

            df = pd.DataFrame(flattened_data)
            df = self._convert_player_profile_types(df)
            
            self.logger.info(f"Extracted player profile data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract player profile data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def extract_player_performance(self,player_ids = None,tours = None,seasons = None):
        self.logger.info("Extracting player performance data")

        query = {}
        if player_ids:
            query["player_id"] = {"$in": player_ids}
        
        try:
            profile_raw = self.db_manager.run_query("player_profile_overview", query)
            
            if not profile_raw:
                self.logger.warning("No player profile data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found profile data for {len(profile_raw)} players")
            flattened_perf_data = []
            
            for profile_doc in profile_raw:
                player_id = profile_doc.get("player_id")
                player_name = f"{profile_doc.get('first_name', '')} {profile_doc.get('last_name', '')}".strip()
                country = profile_doc.get("country")

                performance_data = profile_doc.get("performance", [])
                
                for perf in performance_data:
                    tour = perf.get("tour")
                    season = perf.get("season")
                    display_season = perf.get("display_season")
                    if tours and tour not in tours:
                        continue
                        
                    if seasons and season not in seasons:
                        continue

                    perf_entry = {
                        "player_id": player_id,
                        "player_name": player_name,
                        "country": country,
                        "tour": tour,
                        "season": season,
                        "display_season": display_season
                    }

                    stats = perf.get("stats", [])
                    for stat in stats:
                        title = stat.get("title", "").lower().replace(" ", "_")
                        value = stat.get("value")
                        career = stat.get("career")
                        perf_entry[f"{title}"] = value
                        perf_entry[f"{title}_career"] = career
                    
                    flattened_perf_data.append(perf_entry)
            
            if not flattened_perf_data:
                self.logger.warning("No performance data found after filtering")
                return pd.DataFrame()

            df = pd.DataFrame(flattened_perf_data)
            df = self._convert_player_performance_types(df)
            
            self.logger.info(f"Extracted player performance data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract player performance data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _flatten_player_profile(self, profile_raw):
        flattened_data = []
        
        for profile_doc in profile_raw:
            player_data = {
                "player_id": profile_doc.get("player_id"),
                "first_name": profile_doc.get("first_name"),
                "last_name": profile_doc.get("last_name"),
                "country": profile_doc.get("country"),
                "collected_at": profile_doc.get("collected_at")
            }
            
            # Create full name
            player_data["full_name"] = f"{player_data['first_name']} {player_data['last_name']}".strip()
            
            # Process standings data
            standings = profile_doc.get("standings", {})
            if standings:
                for key, value in standings.items():
                    if key not in ['title', 'description', 'detail_copy']:
                        player_data[f"standings_{key}"] = value
            
            # Process FedEx Fall standings
            fedex_fall = profile_doc.get("fedex_fall_standings", {})
            if fedex_fall:
                for key, value in fedex_fall.items():
                    if key not in ['title', 'description', 'detail_copy']:
                        player_data[f"fedex_fall_{key}"] = value
            
            # Process snapshot data
            snapshot = profile_doc.get("snapshot", [])
            for item in snapshot:
                title = item.get("title", "").lower().replace(" ", "_")
                value = item.get("value")
                description = item.get("description")
                
                player_data[f"snapshot_{title}"] = value
                if description:
                    player_data[f"snapshot_{title}_desc"] = description
            
            # Add latest performance data
            # We'll extract the most recent PGA Tour (R) season if available
            performance = profile_doc.get("performance", [])
            pga_perf = [p for p in performance if p.get("tour") == "R"]
            
            if pga_perf:
                # Sort by season (descending)
                sorted_perf = sorted(pga_perf, key=lambda p: p.get("season", "0"), reverse=True)
                latest_perf = sorted_perf[0]
                
                player_data["latest_season"] = latest_perf.get("season")
                player_data["latest_display_season"] = latest_perf.get("display_season")
                
                # Process latest stats
                stats = latest_perf.get("stats", [])
                for stat in stats:
                    title = stat.get("title", "").lower().replace(" ", "_")
                    value = stat.get("value")
                    player_data[f"latest_{title}"] = value
            
            flattened_data.append(player_data)
        
        return flattened_data

    def _convert_player_profile_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert player profile columns to appropriate data types.
        
        Args:
            df: DataFrame with player profile data
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # String columns to keep as is
        string_cols = [
            'player_id', 'first_name', 'last_name', 'full_name', 'country', 
            'collected_at', 'latest_display_season'
        ]
        
        # Money columns
        money_cols = [col for col in df_converted.columns if any(term in col.lower() for term in ['earnings', 'money', 'purse'])]
        
        # Handle rank columns
        rank_cols = [col for col in df_converted.columns if 'rank' in col.lower()]
        for col in rank_cols:
            if col in df_converted.columns:
                # Handle dash placeholders
                df_converted[col] = df_converted[col].replace('-', None)
                # Convert to numeric
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Convert numeric columns
        for col in df_converted.columns:
            # Skip string columns
            if col in string_cols:
                continue
            
            try:
                # Handle money columns
                if col in money_cols:
                    if df_converted[col].dtype == 'object':
                        # Replace placeholders
                        df_converted[col] = df_converted[col].replace('-', None)
                        # Remove $ and commas, then convert to float
                        df_converted[col] = df_converted[col].astype(str).str.replace('[\\$,]', '', regex=True)
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Try general numeric conversion for everything else
                elif not col.endswith('_desc'):  # Skip description columns
                    # Replace placeholders
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].replace('-', None)
                    
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            except Exception as e:
                # If conversion fails, leave as is
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted

    def _convert_player_performance_types(self, df):
        df_converted = df.copy()
        
        string_cols = [
            'player_id', 'player_name', 'country', 'tour', 'display_season'
        ]
        
        money_cols = [col for col in df_converted.columns if 'earnings' in col.lower()]

        for col in df_converted.columns:
            if col in string_cols:
                continue
            
            try:
                if col in money_cols:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].replace('-', None)
                        df_converted[col] = df_converted[col].astype(str).str.replace('[\\$,]', '', regex=True)
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                elif col == 'season':
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')

                else:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].replace('-', None)
                    
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            except Exception as e:

                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted
    
    
    
    #COURSE_STATS EXTRACTIN
    def extract_course_stats(self,tournament_ids = None,course_ids = None,years= None):
    
        self.logger.info("Extracting course statistics")
        
        # Build the query based on parameters
        query = {}
        
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["tournament_id"] = tournament_ids
            else:
                query["tournament_id"] = {"$in": tournament_ids}
        
        if course_ids:
            if isinstance(course_ids, str):
                query["course_id"] = course_ids
            else:
                query["course_id"] = {"$in": course_ids}
        
        # Extract year from tournament_id if years is specified
        if years:
            # Tournament IDs are in format RYYYY###, so we need to filter
            # We'll do this manually after extraction since MongoDB can't easily query substrings
            pass
        
        try:
            # Get the raw data from MongoDB
            course_raw = self.db_manager.run_query("course_stats", query)
            
            if not course_raw:
                self.logger.warning("No course stats found for the given criteria")
                return pd.DataFrame()
            
            # Apply year filter if specified
            if years:
                if isinstance(years, int):
                    years = [years]
                
                # Filter based on tournament_id containing the year
                filtered_course_raw = []
                for doc in course_raw:
                    tournament_id = doc.get("tournament_id", "")
                    # Extract year from tournament_id (e.g., R2023016 -> 2023)
                    if len(tournament_id) >= 5 and tournament_id[0] == 'R':
                        try:
                            doc_year = int(tournament_id[1:5])
                            if doc_year in years:
                                filtered_course_raw.append(doc)
                        except ValueError:
                            # Skip if we can't parse the year
                            continue
                    else:
                        # Keep the doc if we can't determine the year
                        filtered_course_raw.append(doc)
                
                course_raw = filtered_course_raw
            
            self.logger.info(f"Found stats for {len(course_raw)} course-tournaments")
            
            # Process and flatten the data
            flattened_data = self._flatten_course_stats(course_raw)
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Apply data type conversions
            df = self._convert_course_stats_types(df)
            
            self.logger.info(f"Extracted course stats with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract course stats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def extract_hole_stats(self,
                        tournament_ids: Optional[Union[str, List[str]]] = None,
                        course_ids: Optional[Union[str, List[str]]] = None,
                        round_numbers: Optional[Union[int, List[int]]] = None,
                        hole_numbers: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Extract hole-level statistics from the course_stats collection.
        
        Args:
            tournament_ids: Tournament ID(s) to extract
            course_ids: Course ID(s) to extract
            round_numbers: Round number(s) to filter by
            hole_numbers: Hole number(s) to filter by
            
        Returns:
            DataFrame with hole-level statistics
        """
        self.logger.info("Extracting hole-level statistics")
        
        # Build the query based on parameters
        query = {}
        
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["tournament_id"] = tournament_ids
            else:
                query["tournament_id"] = {"$in": tournament_ids}
        
        if course_ids:
            if isinstance(course_ids, str):
                query["course_id"] = course_ids
            else:
                query["course_id"] = {"$in": course_ids}
        
        try:
            # Get the raw data from MongoDB
            course_raw = self.db_manager.run_query("course_stats", query)
            
            if not course_raw:
                self.logger.warning("No course stats found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found stats for {len(course_raw)} course-tournaments")
            
            # Process and flatten the hole-level data
            flattened_data = self._flatten_hole_stats(
                course_raw, 
                round_numbers=round_numbers,
                hole_numbers=hole_numbers
            )
            
            if not flattened_data:
                self.logger.warning("No hole stats found after filtering")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Apply data type conversions
            df = self._convert_hole_stats_types(df)
            
            self.logger.info(f"Extracted hole stats with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract hole stats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _flatten_course_stats(self, course_raw: List[Dict]) -> List[Dict]:
        """
        Flatten the course statistics data structure.
        
        Args:
            course_raw: List of course stats documents from MongoDB
            
        Returns:
            List of flattened dictionaries
        """
        flattened_data = []
        
        for course_doc in course_raw:
            # Create a baseline dictionary with main fields
            course_data = {
                "tournament_id": course_doc.get("tournament_id"),
                "course_id": course_doc.get("course_id"),
                "course_name": course_doc.get("course_name"),
                "course_code": course_doc.get("course_code"),
                "par": course_doc.get("par"),
                "yardage": course_doc.get("yardage"),
                "host_course": course_doc.get("host_course"),
                "collected_at": course_doc.get("collected_at")
            }
            
            # Extract year from tournament_id if possible
            tournament_id = course_doc.get("tournament_id", "")
            if len(tournament_id) >= 5 and tournament_id[0] == 'R':
                try:
                    course_data["year"] = int(tournament_id[1:5])
                except ValueError:
                    pass
            
            # Process overview data if it exists
            overview = course_doc.get("overview", {})
            if overview:
                # Add basic overview fields
                course_data["overview_name"] = overview.get("name")
                course_data["overview_city"] = overview.get("city")
                course_data["overview_state"] = overview.get("state")
                course_data["overview_country"] = overview.get("country")
                
                # Process overview details
                details = overview.get("details", [])
                for detail in details:
                    label = detail.get("label", "").lower().replace(" ", "_")
                    value = detail.get("value")
                    detail_value = detail.get("detail")
                    
                    course_data[f"overview_{label}"] = value
                    if detail_value:
                        course_data[f"overview_{label}_detail"] = detail_value
            
            # Process course summary data if it exists
            summary = course_doc.get("course_summary", {})
            if summary:
                # Add top-level summary stats
                for key, value in summary.items():
                    if key != "rounds_summary":  # Handle rounds_summary separately
                        course_data[f"summary_{key}"] = value
                
                # Process rounds summary if it exists
                rounds_summary = summary.get("rounds_summary", {})
                for round_num, round_data in rounds_summary.items():
                    for key, value in round_data.items():
                        if key != "header":  # Skip the header field
                            course_data[f"summary_round{round_num}_{key}"] = value
            
            flattened_data.append(course_data)
        
        return flattened_data

    def _flatten_hole_stats(self, 
                        course_raw: List[Dict], 
                        round_numbers: Optional[Union[int, List[int]]] = None,
                        hole_numbers: Optional[Union[int, List[int]]] = None) -> List[Dict]:
        """
        Flatten the hole-level statistics data structure.
        
        Args:
            course_raw: List of course stats documents from MongoDB
            round_numbers: Round number(s) to filter by
            hole_numbers: Hole number(s) to filter by
            
        Returns:
            List of flattened dictionaries
        """
        flattened_data = []
        
        # Normalize filter parameters
        if round_numbers and isinstance(round_numbers, int):
            round_numbers = [round_numbers]
        
        if hole_numbers and isinstance(hole_numbers, int):
            hole_numbers = [hole_numbers]
        
        for course_doc in course_raw:
            # Base course information
            course_base = {
                "tournament_id": course_doc.get("tournament_id"),
                "course_id": course_doc.get("course_id"),
                "course_name": course_doc.get("course_name"),
                "course_code": course_doc.get("course_code"),
                "par": course_doc.get("par"),
                "collected_at": course_doc.get("collected_at")
            }
            
            # Extract year from tournament_id if possible
            tournament_id = course_doc.get("tournament_id", "")
            if len(tournament_id) >= 5 and tournament_id[0] == 'R':
                try:
                    course_base["year"] = int(tournament_id[1:5])
                except ValueError:
                    pass
            
            # Process rounds data
            rounds_data = course_doc.get("rounds", [])
            
            for round_data in rounds_data:
                round_number = round_data.get("round_number")
                
                # Apply round filter if specified
                if round_numbers and round_number not in round_numbers:
                    continue
                
                # Round-specific information
                round_base = {
                    **course_base,
                    "round_number": round_number,
                    "round_header": round_data.get("round_header"),
                    "live": round_data.get("live")
                }
                
                # Process holes data
                holes_data = round_data.get("holes", [])
                
                for hole_data in holes_data:
                    hole_number = hole_data.get("hole_number")
                    
                    # Apply hole filter if specified
                    if hole_numbers and hole_number not in hole_numbers:
                        continue
                    
                    # Create hole-specific entry
                    hole_entry = {
                        **round_base,
                        "hole_number": hole_number,
                        "hole_par": hole_data.get("par"),
                        "hole_yards": hole_data.get("yards"),
                        "hole_scoring_average": hole_data.get("scoring_average"),
                        "hole_scoring_average_diff": hole_data.get("scoring_average_diff"),
                        "hole_scoring_diff_tendency": hole_data.get("scoring_diff_tendency"),
                        "hole_eagles": hole_data.get("eagles"),
                        "hole_birdies": hole_data.get("birdies"),
                        "hole_pars": hole_data.get("pars"),
                        "hole_bogeys": hole_data.get("bogeys"),
                        "hole_double_bogeys": hole_data.get("double_bogeys"),
                        "hole_rank": hole_data.get("rank"),
                        "hole_live": hole_data.get("live"),
                        "hole_about": hole_data.get("about_this_hole")
                    }
                    
                    # Process pin location if it exists
                    pin_location = hole_data.get("pin_location", {})
                    if pin_location:
                        left_to_right = pin_location.get("left_to_right", {})
                        if left_to_right:
                            hole_entry["pin_left_to_right_x"] = left_to_right.get("x")
                            hole_entry["pin_left_to_right_y"] = left_to_right.get("y")
                            hole_entry["pin_left_to_right_z"] = left_to_right.get("z")
                        
                        bottom_to_top = pin_location.get("bottom_to_top", {})
                        if bottom_to_top:
                            hole_entry["pin_bottom_to_top_x"] = bottom_to_top.get("x")
                            hole_entry["pin_bottom_to_top_y"] = bottom_to_top.get("y")
                            hole_entry["pin_bottom_to_top_z"] = bottom_to_top.get("z")
                    
                    flattened_data.append(hole_entry)
        
        return flattened_data

    def _convert_course_stats_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert course stats columns to appropriate data types.
        
        Args:
            df: DataFrame with course stats
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # String columns to keep as is
        string_cols = [
            'tournament_id', 'course_id', 'course_name', 'course_code', 'collected_at',
            'overview_name', 'overview_city', 'overview_state', 'overview_country'
        ]
        
        # Overview detail columns that should stay as strings
        overview_string_cols = [
            'overview_fairway', 'overview_rough', 'overview_green', 
            'overview_design', 'overview_record_detail', 'overview_established'
        ]
        string_cols.extend([col for col in df_converted.columns if col in overview_string_cols])
        
        # Boolean columns
        bool_cols = ['host_course']
        
        # Convert boolean columns
        for col in bool_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(bool)
        
        # Convert integer columns
        int_cols = ['par', 'year']
        for col in int_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
        
        # Convert numeric columns
        for col in df_converted.columns:
            # Skip string and bool columns
            if col in string_cols or col in bool_cols:
                continue
            
            try:
                # Try general numeric conversion
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except Exception as e:
                # If conversion fails, leave as is
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted

    def _convert_hole_stats_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert hole stats columns to appropriate data types.
        
        Args:
            df: DataFrame with hole stats
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # String columns to keep as is
        string_cols = [
            'tournament_id', 'course_id', 'course_name', 'course_code', 
            'collected_at', 'round_header', 'hole_scoring_diff_tendency',
            'hole_about'
        ]
        
        # Boolean columns
        bool_cols = ['live', 'hole_live']
        
        # Convert boolean columns
        for col in bool_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(bool)
        
        # Convert integer columns
        int_cols = ['par', 'year', 'round_number', 'hole_number', 'hole_par', 
                    'hole_eagles', 'hole_birdies', 'hole_pars', 'hole_bogeys', 
                    'hole_double_bogeys', 'hole_rank']
        
        for col in int_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
        
        # Convert numeric columns
        for col in df_converted.columns:
            # Skip string and bool columns
            if col in string_cols or col in bool_cols:
                continue
            
            # Skip already converted integer columns
            if col in int_cols:
                continue
            
            try:
                # Try general numeric conversion
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except Exception as e:
                # If conversion fails, leave as is
                self.logger.debug(f"Could not convert column {col}: {str(e)}")
                continue
        
        return df_converted
    




## current form data extraction
    def extract_current_form(self, 
                            tournament_id: Optional[str] = None,
                            player_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract current form data from the current_form collection.
        
        Args:
            tournament_id: Tournament ID to filter by
            player_ids: List of player IDs to filter by
            
        Returns:
            DataFrame with player current form data
        """
        self.logger.info("Extracting current form data")
        
        # Build the query based on parameters
        query = {"field_stat_type": "CURRENT_FORM"}
        
        if tournament_id:
            query["tournament_id"] = tournament_id
        
        try:
            # Get the raw data from MongoDB
            current_form_raw = self.db_manager.run_query("current_form", query)
            
            if not current_form_raw:
                self.logger.warning("No current form data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(current_form_raw)} current form documents")
            
            # Process and flatten the data
            flattened_data = self._flatten_current_form(current_form_raw, player_ids)
            
            if not flattened_data:
                self.logger.warning("No current form data after filtering by player IDs")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Apply data type conversions
            df = self._convert_current_form_types(df)
            
            self.logger.info(f"Extracted current form data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract current form data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _flatten_current_form(self, 
                             current_form_raw: List[Dict], 
                             player_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Flatten the nested current form structure into a list of dictionaries.
        
        Args:
            current_form_raw: List of current form documents from MongoDB
            player_ids: Optional list of player IDs to filter by
            
        Returns:
            List of flattened dictionaries
        """
        flattened_data = []
        
        for form_doc in current_form_raw:
            tournament_id = form_doc.get("tournament_id")
            collected_at = form_doc.get("collected_at")
            
            # Get stat headers
            sg_headers = form_doc.get("strokes_gained_header", [])
            
            # Process each player
            players_data = form_doc.get("players", [])
            
            for player in players_data:
                player_id = player.get("player_id")
                
                # Skip if not in requested player IDs
                if player_ids and player_id not in player_ids:
                    continue
                
                # Basic player info
                player_record = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "total_rounds": player.get("total_rounds"),
                    "collected_at": collected_at
                }
                
                # Process tournament results
                tournament_results = player.get("tournament_results", [])
                
                # Add the last 5 tournament results
                for i, result in enumerate(tournament_results):
                    result_num = i + 1
                    tournament_prefix = f"last{result_num}"
                    
                    player_record[f"{tournament_prefix}_tournament_id"] = result.get("tournament_id")
                    player_record[f"{tournament_prefix}_tournament_name"] = result.get("name")
                    player_record[f"{tournament_prefix}_end_date"] = result.get("end_date")
                    player_record[f"{tournament_prefix}_position"] = result.get("position")
                    player_record[f"{tournament_prefix}_score"] = result.get("score")
                    player_record[f"{tournament_prefix}_season"] = result.get("season")
                    player_record[f"{tournament_prefix}_tour_code"] = result.get("tour_code")
                
                # Process strokes gained
                strokes_gained = player.get("strokes_gained", [])
                
                # Map strokes gained to their headers
                for i, sg in enumerate(strokes_gained):
                    if i < len(sg_headers):
                        # Clean up the header for use as a column name
                        header = sg_headers[i].replace("SG: ", "sg_").lower()
                        
                        player_record[f"{header}_value"] = sg.get("stat_value")
                        player_record[f"{header}_color"] = sg.get("stat_color")
                        player_record[f"{header}_id"] = sg.get("stat_id")
                
                flattened_data.append(player_record)
        
        return flattened_data
    
    def _convert_current_form_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert current form columns to appropriate data types.
        
        Args:
            df: DataFrame with current form data
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # Convert numeric columns
        numeric_columns = ["total_rounds"]
        
        # Add dynamic columns for scores and seasons
        score_columns = [col for col in df_converted.columns if col.endswith("_score")]
        season_columns = [col for col in df_converted.columns if col.endswith("_season")]
        sg_value_columns = [col for col in df_converted.columns if col.endswith("_value")]
        
        numeric_columns.extend(score_columns)
        numeric_columns.extend(season_columns)
        numeric_columns.extend(sg_value_columns)
        
        for col in numeric_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Convert date columns
        date_columns = [col for col in df_converted.columns if col.endswith("_end_date")]
        
        for col in date_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
        
        return df_converted

    def extract_course_fit(self, 
                         tournament_id: Optional[str] = None,
                         player_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract course fit data from the field_stats collection.
        
        Args:
            tournament_id: Tournament ID to filter by
            player_ids: List of player IDs to filter by
            
        Returns:
            DataFrame with player course fit data
        """
        self.logger.info("Extracting course fit data")
        
        # Build the query based on parameters
        query = {"field_stat_type": "COURSE_FIT"}
        
        if tournament_id:
            query["tournament_id"] = tournament_id
        
        try:
            # Get the raw data from MongoDB
            course_fit_raw = self.db_manager.run_query("field_stats", query)
            
            if not course_fit_raw:
                self.logger.warning("No course fit data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(course_fit_raw)} course fit documents")
            
            # Process and flatten the data
            flattened_data = self._flatten_course_fit(course_fit_raw, player_ids)
            
            if not flattened_data:
                self.logger.warning("No course fit data after filtering by player IDs")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Apply data type conversions
            df = self._convert_course_fit_types(df)
            
            self.logger.info(f"Extracted course fit data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract course fit data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _flatten_course_fit(self, 
                          course_fit_raw: List[Dict], 
                          player_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Flatten the nested course fit structure into a list of dictionaries.
        
        Args:
            course_fit_raw: List of course fit documents from MongoDB
            player_ids: Optional list of player IDs to filter by
            
        Returns:
            List of flattened dictionaries
        """
        flattened_data = []
        
        for fit_doc in course_fit_raw:
            tournament_id = fit_doc.get("tournament_id")
            collected_at = fit_doc.get("collected_at")
            
            # Get stat headers
            stat_headers = fit_doc.get("stat_headers", [])
            
            # Process each player
            players_data = fit_doc.get("players", [])
            
            for player in players_data:
                player_id = player.get("player_id")
                
                # Skip if not in requested player IDs
                if player_ids and player_id not in player_ids:
                    continue
                
                # Basic player info
                player_record = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "total_rounds": player.get("total_rounds"),
                    "score": player.get("score"),
                    "collected_at": collected_at
                }
                
                # Process stats
                stats = player.get("stats", [])
                
                # Map stats to their headers
                for i, stat in enumerate(stats):
                    header = stat.get("header", "")
                    if not header and i < len(stat_headers):
                        header = stat_headers[i]
                    
                    if header:
                        # Clean up the header for use as a column name
                        column_name = header.lower().replace(": ", "_").replace(" - ", "_").replace(" ", "_")
                        
                        player_record[f"{column_name}_value"] = stat.get("value")
                        player_record[f"{column_name}_rank"] = stat.get("rank")
                        player_record[f"{column_name}_color"] = stat.get("color")
                
                flattened_data.append(player_record)
        
        return flattened_data
    
    def _convert_course_fit_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_converted = df.copy()
        
        # Convert numeric columns
        numeric_columns = ["total_rounds", "score"]
        
        # Add dynamic columns for stat values and ranks
        value_columns = [col for col in df_converted.columns if col.endswith("_value")]
        rank_columns = [col for col in df_converted.columns if col.endswith("_rank")]
        
        numeric_columns.extend(value_columns)
        numeric_columns.extend(rank_columns)
        
        for col in numeric_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        return df_converted

    def extract_tournament_history_stats(self,  tournament_id: Optional[str] = None,player_ids: Optional[List[str]] = None) -> pd.DataFrame:
        self.logger.info("Extracting tournament history stats")
        
        # Build the query based on parameters
        query = {"field_stat_type": "TOURNAMENT_HISTORY"}
        
        if tournament_id:
            query["tournament_id"] = tournament_id
        
        try:
            # Get the raw data from MongoDB - using correct collection name
            history_raw = self.db_manager.run_query("tournament_history_stats", query)
            
            if not history_raw:
                self.logger.warning("No tournament history stats found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(history_raw)} tournament history documents")
            
            # Process and flatten the data
            flattened_data = self._flatten_tournament_history_stats(history_raw, player_ids)
            
            if not flattened_data:
                self.logger.warning("No tournament history stats after filtering by player IDs")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Apply data type conversions
            df = self._convert_tournament_history_stats_types(df)
            
            self.logger.info(f"Extracted tournament history stats with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract tournament history stats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _flatten_tournament_history_stats(self, 
                                       history_raw: List[Dict], 
                                       player_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Flatten the nested tournament history structure into a list of dictionaries.
        
        Args:
            history_raw: List of tournament history documents from MongoDB
            player_ids: Optional list of player IDs to filter by
            
        Returns:
            List of flattened dictionaries
        """
        flattened_data = []
        
        for history_doc in history_raw:
            tournament_id = history_doc.get("tournament_id")
            collected_at = history_doc.get("collected_at")
            
            # Get top-level stat headers (fallback)
            doc_sg_headers = history_doc.get("stat_headers", [])
            
            # Process each player
            players_data = history_doc.get("players", [])
            
            for player in players_data:
                player_id = player.get("player_id")
                
                # Skip if not in requested player IDs
                if player_ids and player_id not in player_ids:
                    continue
                
                # Basic player info
                player_record = {
                    "tournament_id": tournament_id,
                    "player_id": player_id,
                    "total_rounds": player.get("total_rounds"),
                    "collected_at": collected_at
                }
                
                # Process tournament results
                tournament_results = player.get("tournament_results", [])
                
                # Add the tournament results
                for i, result in enumerate(tournament_results):
                    result_num = i + 1
                    history_prefix = f"history{result_num}"
                    
                    player_record[f"{history_prefix}_tournament_id"] = result.get("tournament_id")
                    player_record[f"{history_prefix}_tournament_name"] = result.get("name")
                    player_record[f"{history_prefix}_end_date"] = result.get("end_date")
                    player_record[f"{history_prefix}_position"] = result.get("position")
                    player_record[f"{history_prefix}_score"] = result.get("score")
                    player_record[f"{history_prefix}_season"] = result.get("season")
                    player_record[f"{history_prefix}_tour_code"] = result.get("tour_code")
                
                # Process strokes gained
                strokes_gained = player.get("strokes_gained", [])
                
                # Try to get headers from player first, fall back to document headers
                sg_headers = player.get("strokes_gained_header", doc_sg_headers)
                
                # Map strokes gained to their headers
                for i, sg in enumerate(strokes_gained):
                    if i < len(sg_headers):
                        # Clean up the header for use as a column name
                        header = sg_headers[i].replace("SG: ", "sg_").lower()
                        
                        player_record[f"{header}_value"] = sg.get("stat_value")
                        player_record[f"{header}_color"] = sg.get("stat_color")
                        player_record[f"{header}_id"] = sg.get("stat_id")
                
                flattened_data.append(player_record)
        
        return flattened_data
    
    def _convert_tournament_history_stats_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert tournament history stats columns to appropriate data types.
        
        Args:
            df: DataFrame with tournament history stats
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # Convert numeric columns
        numeric_columns = ["total_rounds"]
        
        # Add dynamic columns for scores and seasons
        score_columns = [col for col in df_converted.columns if col.endswith("_score")]
        season_columns = [col for col in df_converted.columns if col.endswith("_season")]
        sg_value_columns = [col for col in df_converted.columns if col.endswith("_value")]
        
        numeric_columns.extend(score_columns)
        numeric_columns.extend(season_columns)
        numeric_columns.extend(sg_value_columns)
        
        for col in numeric_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Convert date columns
        date_columns = [col for col in df_converted.columns if col.endswith("_end_date")]
        
        for col in date_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
        
        return df_converted
    
    
    
    
    
    # SCORECARD DATA EXTRACTION
    def extract_player_scorecards(self,
                             tournament_ids: Optional[Union[str, List[str]]] = None,
                             player_ids: Optional[Union[str, List[str]]] = None,
                             round_numbers: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Extract player scorecard data from the scorecards collection.
        
        Args:
            tournament_ids: Tournament ID(s) to extract
            player_ids: Player ID(s) to extract
            round_numbers: Round number(s) to filter by
            
        Returns:
            DataFrame with player scorecard data
        """
        self.logger.info("Extracting player scorecard data")
        
        # Build the query based on parameters
        query = {}
        
        if tournament_ids:
            # Handle tournament ID format in the database
            if isinstance(tournament_ids, str):
                # Look for IDs that start with the tournament ID followed by a dash
                query["id"] = {"$regex": f"^{tournament_ids}-"}
            else:
                # If multiple tournament IDs, create a regex pattern for each
                tournament_patterns = [f"^{t_id}-" for t_id in tournament_ids]
                query["id"] = {"$regex": "|".join(tournament_patterns)}
        
        if player_ids:
            if isinstance(player_ids, str):
                # Extract player_id from database id format (e.g., "R2024016-33948")
                query["player_id"] = player_ids
            else:
                query["player_id"] = {"$in": player_ids}
        
        try:
            # Get the raw data from MongoDB
            scorecards_raw = self.db_manager.run_query("scorecards", query)
            
            if not scorecards_raw:
                self.logger.warning("No scorecard data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(scorecards_raw)} player scorecards")
            
            # Filter round data if round_numbers is specified
            if round_numbers:
                round_data = self._flatten_player_scorecards_by_round(
                    scorecards_raw, 
                    round_numbers=round_numbers
                )
            else:
                # Get all rounds in a flattened structure
                round_data = self._flatten_player_scorecards_by_round(scorecards_raw)
            
            if not round_data:
                self.logger.warning(f"No scorecard data found after filtering by rounds")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(round_data)
            
            # Apply data type conversions
            df = self._convert_scorecard_types(df)
            
            self.logger.info(f"Extracted scorecard data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract scorecard data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def extract_player_hole_scores(self, tournament_ids=None,player_ids=None,round_numbers=None,hole_numbers=None):

        self.logger.info("Extracting hole-by-hole scorecard data")

        query = {}
        
        if tournament_ids:
            if isinstance(tournament_ids, str):
                query["id"] = {"$regex": f"^{tournament_ids}-"}
            else:
                # If multiple tournament IDs, create a regex pattern for each
                tournament_patterns = [f"^{t_id}-" for t_id in tournament_ids]
                query["id"] = {"$regex": "|".join(tournament_patterns)}
        
        if player_ids:
            if isinstance(player_ids, str):
                query["player_id"] = player_ids
            else:
                query["player_id"] = {"$in": player_ids}
        
        try:
            scorecards_raw = self.db_manager.run_query("scorecards", query)
            
            if not scorecards_raw:
                self.logger.warning("No scorecard data found for the given criteria")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(scorecards_raw)} player scorecards")
            
            # Get hole-by-hole data in a flattened structure
            hole_data = self._flatten_player_scorecards_by_hole(
                scorecards_raw, 
                round_numbers=round_numbers,
                hole_numbers=hole_numbers
            )
            
            if not hole_data:
                self.logger.warning(f"No hole-by-hole data found after filtering")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(hole_data)
            
            # Apply data type conversions
            df = self._convert_hole_score_types(df)
            
            self.logger.info(f"Extracted hole-by-hole data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract hole-by-hole data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _flatten_player_scorecards_by_round(self, 
                                        scorecards_raw: List[Dict],
                                        round_numbers: Optional[Union[int, List[int]]] = None) -> List[Dict]:
        """
        Flatten the player scorecard data structure by round.
        
        Args:
            scorecards_raw: List of scorecard documents from MongoDB
            round_numbers: Round number(s) to filter by
            
        Returns:
            List of flattened dictionaries, one per player-round
        """
        flattened_data = []
        
        # Normalize round_numbers to a list if provided
        if round_numbers and isinstance(round_numbers, int):
            round_numbers = [round_numbers]
        
        for scorecard_doc in scorecards_raw:
            # Extract base tournament and player info
            doc_id = scorecard_doc.get("id", "")
            tournament_id = doc_id.split("-")[0] if "-" in doc_id else None
            
            # Base player info
            player_base = {
                "tournament_id": tournament_id,
                "tournament_name": scorecard_doc.get("tournament_name"),
                "player_id": scorecard_doc.get("player_id"),
                "player_name": scorecard_doc.get("player_name"),
                "player_country": scorecard_doc.get("player_country"),
                "current_round": scorecard_doc.get("currentRound"),
                "player_state": scorecard_doc.get("playerState"),
                "collected_at": scorecard_doc.get("collected_at")
            }
            
            # Process each round
            round_scores = scorecard_doc.get("roundScores", [])
            
            for round_data in round_scores:
                round_number = round_data.get("roundNumber")
                
                # Apply round filter if specified
                if round_numbers and round_number not in round_numbers:
                    continue
                
                # Create round-specific entry
                round_entry = {
                    **player_base,
                    "round_number": round_number,
                    "complete": round_data.get("complete"),
                    "current_hole": round_data.get("currentHole"),
                    "is_current_round": round_data.get("currentRound"),
                    "course_name": round_data.get("courseName"),
                    "par_total": round_data.get("parTotal"),
                    "round_total": round_data.get("total"),
                    "score_to_par": round_data.get("scoreToPar")
                }
                
                # Add front nine summary if available
                first_nine = round_data.get("firstNine", {})
                if first_nine:
                    round_entry["front_nine_total"] = first_nine.get("total")
                    round_entry["front_nine_par"] = first_nine.get("parTotal")
                
                # Add back nine summary if available
                second_nine = round_data.get("secondNine", {})
                if second_nine:
                    round_entry["back_nine_total"] = second_nine.get("total")
                    round_entry["back_nine_par"] = second_nine.get("parTotal")
                
                flattened_data.append(round_entry)
        
        return flattened_data

    def _flatten_player_scorecards_by_hole(self, 
                                        scorecards_raw: List[Dict],
                                        round_numbers: Optional[Union[int, List[int]]] = None,
                                        hole_numbers: Optional[Union[int, List[int]]] = None) -> List[Dict]:
        """
        Flatten the player scorecard data structure by hole.
        
        Args:
            scorecards_raw: List of scorecard documents from MongoDB
            round_numbers: Round number(s) to filter by
            hole_numbers: Hole number(s) to filter by
            
        Returns:
            List of flattened dictionaries, one per player-round-hole
        """
        flattened_data = []
        
        # Normalize filter parameters
        if round_numbers and isinstance(round_numbers, int):
            round_numbers = [round_numbers]
        
        if hole_numbers and isinstance(hole_numbers, int):
            hole_numbers = [hole_numbers]
        
        for scorecard_doc in scorecards_raw:
            # Extract base tournament and player info
            doc_id = scorecard_doc.get("id", "")
            tournament_id = doc_id.split("-")[0] if "-" in doc_id else None
            
            # Base player info
            player_base = {
                "tournament_id": tournament_id,
                "tournament_name": scorecard_doc.get("tournament_name"),
                "player_id": scorecard_doc.get("player_id"),
                "player_name": scorecard_doc.get("player_name"),
                "player_country": scorecard_doc.get("player_country"),
                "collected_at": scorecard_doc.get("collected_at")
            }
            
            # Process each round
            round_scores = scorecard_doc.get("roundScores", [])
            
            for round_data in round_scores:
                round_number = round_data.get("roundNumber")
                
                # Apply round filter if specified
                if round_numbers and round_number not in round_numbers:
                    continue
                
                # Round info
                round_base = {
                    **player_base,
                    "round_number": round_number,
                    "complete": round_data.get("complete"),
                    "is_current_round": round_data.get("currentRound"),
                    "course_name": round_data.get("courseName"),
                    "par_total": round_data.get("parTotal"),
                    "round_total": round_data.get("total"),
                    "score_to_par": round_data.get("scoreToPar")
                }
                
                # Process front nine holes
                first_nine = round_data.get("firstNine", {})
                front_nine_holes = first_nine.get("holes", [])
                
                for hole_data in front_nine_holes:
                    hole_number = hole_data.get("holeNumber")
                    
                    # Apply hole filter if specified
                    if hole_numbers and hole_number not in hole_numbers:
                        continue
                    
                    # Create hole-specific entry
                    hole_entry = {
                        **round_base,
                        "hole_number": hole_number,
                        "nine": "FRONT",
                        "hole_par": hole_data.get("par"),
                        "hole_score": hole_data.get("score"),
                        "hole_status": hole_data.get("status"),
                        "hole_yardage": hole_data.get("yardage"),
                        "running_score": hole_data.get("roundScore")
                    }
                    
                    flattened_data.append(hole_entry)
                
                # Process back nine holes
                second_nine = round_data.get("secondNine", {})
                back_nine_holes = second_nine.get("holes", [])
                
                for hole_data in back_nine_holes:
                    hole_number = hole_data.get("holeNumber")
                    
                    # Apply hole filter if specified
                    if hole_numbers and hole_number not in hole_numbers:
                        continue
                    
                    # Create hole-specific entry
                    hole_entry = {
                        **round_base,
                        "hole_number": hole_number,
                        "nine": "BACK",
                        "hole_par": hole_data.get("par"),
                        "hole_score": hole_data.get("score"),
                        "hole_status": hole_data.get("status"),
                        "hole_yardage": hole_data.get("yardage"),
                        "running_score": hole_data.get("roundScore")
                    }
                    
                    flattened_data.append(hole_entry)
        
        return flattened_data

    def _convert_scorecard_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert scorecard columns to appropriate data types.
        
        Args:
            df: DataFrame with scorecard data
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        df_converted = df.copy()
        
        # String columns to keep as is
        string_cols = ['tournament_id', 'tournament_name', 'player_id', 'player_name','player_country', 'player_state', 'course_name', 'collected_at']
        
        # Boolean columns
        bool_cols = ['complete', 'is_current_round']
        
        # Convert boolean columns
        for col in bool_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(bool)
        
        # Convert numeric columns
        numeric_cols = [
            'current_round', 'round_number', 'current_hole', 
            'par_total', 'front_nine_par', 'back_nine_par'
        ]
        
        for col in numeric_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Handle score columns that might be strings (like "68")
        score_cols = ['round_total', 'front_nine_total', 'back_nine_total']
        
        for col in score_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Handle score_to_par which might be like "-5"
        if 'score_to_par' in df_converted.columns:
            # Extract numeric value with sign
            df_converted['score_to_par'] = df_converted['score_to_par'].astype(str).str.replace('E', '0')
            df_converted['score_to_par'] = pd.to_numeric(df_converted['score_to_par'], errors='coerce')
        
        # Convert timestamp
        if 'collected_at' in df_converted.columns:
            try:
                df_converted['collected_at'] = pd.to_datetime(df_converted['collected_at'])
            except:
                pass
        
        return df_converted

    def _convert_hole_score_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_converted = df.copy()

        string_cols = ['tournament_id', 'tournament_name', 'player_id', 'player_name','player_country', 'course_name', 'hole_status', 'nine', 'collected_at' ]

        bool_cols = ['complete', 'is_current_round']

        for col in bool_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(bool)
        int_cols = ['round_number', 'hole_number', 'hole_par', 'hole_yardage']
        
        for col in int_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')

        numeric_cols = ['par_total', 'round_total', 'hole_score']
        
        for col in numeric_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Handle score_to_par and running_score which might be like "-5" or "E"
        for col in ['score_to_par', 'running_score']:
            if col in df_converted.columns:
                # Replace "E" with "0" and convert to numeric
                df_converted[col] = df_converted[col].astype(str).str.replace('E', '0')
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
        # Convert timestamp
        if 'collected_at' in df_converted.columns:
            try:
                df_converted['collected_at'] = pd.to_datetime(df_converted['collected_at'])
            except:
                pass
        
        return df_converted

    def calculate_player_round_stats(self, player_hole_data):
        self.logger.info("Calculating player round statistics")
        
        if player_hole_data.empty:
            self.logger.warning("No hole data provided for stats calculation")
            return pd.DataFrame()
        
        try:
            grouped = player_hole_data.groupby(['player_id', 'tournament_id', 'round_number'])
            
            round_stats = []
            
            for (player_id, tournament_id, round_number), group in grouped:
                first_row = group.iloc[0]
                stats = {
                    'player_id': player_id,
                    'player_name': first_row['player_name'],
                    'tournament_id': tournament_id,
                    'tournament_name': first_row['tournament_name'],
                    'round_number': round_number,
                    'course_name': first_row['course_name'],
                    'par_total': first_row['par_total'],
                    'round_total': first_row['round_total'],
                    'score_to_par': first_row['score_to_par']
                }

                hole_status_counts = group['hole_status'].value_counts()

                stats['eagles'] = hole_status_counts.get('EAGLE', 0)
                stats['birdies'] = hole_status_counts.get('BIRDIE', 0)
                stats['pars'] = hole_status_counts.get('PAR', 0)
                stats['bogeys'] = hole_status_counts.get('BOGEY', 0)
                stats['double_bogeys'] = hole_status_counts.get('DOUBLE BOGEY', 0)
                # Count anything worse than double bogey
                stats['others'] = sum(hole_status_counts.get(status, 0) for status in 
                                    hole_status_counts.index if status not in 
                                    ['EAGLE', 'BIRDIE', 'PAR', 'BOGEY', 'DOUBLE BOGEY'])
                
                # Calculate front nine / back nine totals
                front_nine = group[group['nine'] == 'FRONT']
                back_nine = group[group['nine'] == 'BACK']
                
                if not front_nine.empty:
                    stats['front_nine_score'] = sum(pd.to_numeric(front_nine['hole_score'], errors='coerce'))
                    stats['front_nine_par'] = sum(front_nine['hole_par'])
                    stats['front_nine_to_par'] = stats['front_nine_score'] - stats['front_nine_par']
                
                if not back_nine.empty:
                    stats['back_nine_score'] = sum(pd.to_numeric(back_nine['hole_score'], errors='coerce'))
                    stats['back_nine_par'] = sum(back_nine['hole_par'])
                    stats['back_nine_to_par'] = stats['back_nine_score'] - stats['back_nine_par']
                
                # Calculate par-3, par-4, par-5 performance
                par3_holes = group[group['hole_par'] == 3]
                par4_holes = group[group['hole_par'] == 4]
                par5_holes = group[group['hole_par'] == 5]
                
                if not par3_holes.empty:
                    stats['par3_score'] = sum(pd.to_numeric(par3_holes['hole_score'], errors='coerce'))
                    stats['par3_count'] = len(par3_holes)
                    stats['par3_to_par'] = stats['par3_score'] - (3 * stats['par3_count'])
                    stats['par3_average'] = stats['par3_score'] / stats['par3_count']
                
                if not par4_holes.empty:
                    stats['par4_score'] = sum(pd.to_numeric(par4_holes['hole_score'], errors='coerce'))
                    stats['par4_count'] = len(par4_holes)
                    stats['par4_to_par'] = stats['par4_score'] - (4 * stats['par4_count'])
                    stats['par4_average'] = stats['par4_score'] / stats['par4_count']
                
                if not par5_holes.empty:
                    stats['par5_score'] = sum(pd.to_numeric(par5_holes['hole_score'], errors='coerce'))
                    stats['par5_count'] = len(par5_holes)
                    stats['par5_to_par'] = stats['par5_score'] - (5 * stats['par5_count'])
                    stats['par5_average'] = stats['par5_score'] / stats['par5_count']
                
                round_stats.append(stats)
            
            return pd.DataFrame(round_stats)
        
        except Exception as e:
            self.logger.error(f"Failed to calculate round statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def calculate_tournament_stats(self, tournament_id: str,hole_data: pd.DataFrame = None) -> pd.DataFrame:
        self.logger.info(f"Calculating tournament-wide statistics for {tournament_id}")
        
        try:
            # If hole data wasn't provided, extract it
            if hole_data is None or hole_data.empty:
                hole_data = self.extract_player_hole_scores(tournament_ids=tournament_id)
                
            if hole_data.empty:
                self.logger.warning(f"No hole data found for tournament {tournament_id}")
                return pd.DataFrame()
            
            # Group by round and hole
            grouped = hole_data.groupby(['round_number', 'hole_number'])
            
            hole_stats = []
            
            for (round_number, hole_number), group in grouped:
                # Get hole info from first row
                first_row = group.iloc[0]
                
                # Base stats dictionary
                stats = {
                    'tournament_id': tournament_id,
                    'tournament_name': first_row['tournament_name'],
                    'round_number': round_number,
                    'hole_number': hole_number,
                    'hole_par': first_row['hole_par'],
                    'hole_yardage': first_row['hole_yardage'],
                    'player_count': len(group),
                    'course_name': first_row['course_name']
                }
                
                # Calculate scoring statistics
                hole_scores = pd.to_numeric(group['hole_score'], errors='coerce')
                stats['average_score'] = hole_scores.mean()
                stats['score_stdev'] = hole_scores.std()
                stats['min_score'] = hole_scores.min()
                stats['max_score'] = hole_scores.max()
                stats['score_to_par'] = stats['average_score'] - stats['hole_par']
                
                # Calculate counts of each score type
                hole_status_counts = group['hole_status'].value_counts()
                
                # Add counts and percentages of each type (ensure 0 if none)
                for status in ['EAGLE', 'BIRDIE', 'PAR', 'BOGEY', 'DOUBLE BOGEY']:
                    count = hole_status_counts.get(status, 0)
                    stats[status.lower().replace(' ', '_') + '_count'] = count
                    stats[status.lower().replace(' ', '_') + '_pct'] = count / stats['player_count'] * 100
                
                # Count anything worse than double bogey as "other"
                other_count = sum(hole_status_counts.get(status, 0) for status in 
                                hole_status_counts.index if status not in 
                                ['EAGLE', 'BIRDIE', 'PAR', 'BOGEY', 'DOUBLE BOGEY'])
                stats['other_count'] = other_count
                stats['other_pct'] = other_count / stats['player_count'] * 100
                
                hole_stats.append(stats)
            
            return pd.DataFrame(hole_stats)
        
        except Exception as e:
            self.logger.error(f"Failed to calculate tournament statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
