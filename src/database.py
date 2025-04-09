from pymongo import MongoClient, ASCENDING  # type: ignore
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel  # type: ignore
from typing import Optional

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlayerStat(BaseModel):
    player_id: Optional[str] = None
    name: Optional[str] = None
    season: Optional[str] = None
    stat_id: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    rank: Optional[int] = None
    value: Optional[float] = None
    supporting_stat_description: Optional[str] = None
    supporting_stat_value: Optional[float] = None
    supporting_value_description: Optional[str] = None
    supporting_value: Optional[float] = None
    collected_at: datetime = datetime.utcnow()

class CourseStat(BaseModel):
    tournament_id: Optional[str] = None
    course_id: Optional[str] = None
    holes: Optional[int] = None
    course_summary: Optional[str] = None
    collected_at: datetime = datetime.utcnow()


class QueryCache:
    def __init__(self, maxsize=1000, ttl=3600):
        self.maxsize = maxsize
        self.ttl = timedelta(seconds=ttl)
        self.cache = {}
        
    def get(self, key):
        entry = self.cache.get(key)
        if entry and (datetime.now() - entry['timestamp']) < self.ttl:
            return entry['data']
        return None
        
    def set(self, key, data):
        if len(self.cache) >= self.maxsize:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = {'data': data, 'timestamp': datetime.now()}



#initilise database 
class DatabaseManager:
    def __init__(self, uri: str, database_name: str = "pga_tour_data"):
        self.logger = logging.getLogger(__name__)
        
        self.cache = QueryCache(maxsize=5000, ttl=1800)  # 30 minute TTL
        try:
            self.client = MongoClient(uri, 
                                    maxPoolSize=50,  # Connection pooling
                                    serverSelectionTimeoutMS=5000,  # 5 second timeout
                                    waitQueueTimeoutMS=2500)
            self.db = self.client[database_name]
            self.setup_indexes()
            logger.info(f"Successfully connected to database: {database_name}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def setup_indexes(self):
        try:
            # Player stats indexes
            self.db.player_stats.create_index([("player_id", ASCENDING), ("season", ASCENDING)])
            self.db.player_stats.create_index([("collected_at", -1)])

            # Course stats indexes
            self.db.course_stats.create_index([("tournament_id", ASCENDING), ("course_id", ASCENDING)])
            self.db.course_stats.create_index([("collected_at", -1)])

            # Shot data indexes
            self.db.shot_data.create_index([
                ("tournament_id", ASCENDING),
                ("course_id", ASCENDING)
            ])
            self.db.shot_data.create_index([("collected_at", -1)])
            
            logger.info("Successfully created database indexes")
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    def insert_player_stats(self, stats_data, collection_name = "player_stats"):
        try:
            collection = self.db[collection_name]
            
            # Create indexes if they don't exist
            if 'player_id_1_season_1' not in collection.index_information():
                collection.create_index([("player_id", 1), ("season", 1)], unique=True)
            
            # 
            result = collection.update_one(
                {
                    "player_id": stats_data["player_id"],
                    "season": stats_data["season"]
                },
                {"$set": stats_data},
                upsert=True
            )
            
            logger.info(f"Successfully stored stats for player {stats_data['name']} in season {stats_data['season']}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert player stats: {str(e)}")
            raise

    def insert_course_stats(self, course_stats, collection_name= "course_stats"):
        try:
            collection = self.db[collection_name]
            
            # For debugging - log the rounds data before insertion
            if isinstance(course_stats, dict):
                rounds = course_stats.get('rounds', [])
                self.logger.info(f"Inserting course stats with {len(rounds)} rounds")
                
                # Create a unique key for upsert
                query = {
                    "tournament_id": course_stats.get('tournament_id'),
                    "course_id": course_stats.get('course_id')
                }
                
                # Use replace_one with upsert to ensure we don't duplicate data
                # and preserve the complete data structure including rounds
                result = collection.replace_one(query, course_stats, upsert=True)
                
                self.logger.info(f"Successfully inserted/updated course stats for tournament {course_stats.get('tournament_id')}, course {course_stats.get('course_id')}")
                return result.upserted_id or result.modified_count
                
            elif isinstance(course_stats, list):
                results = []
                for stats in course_stats:
                    result = self.insert_course_stats(stats, collection_name)
                    results.append(result)
                return results
            else:
                self.logger.warning(f"Invalid course stats type: {type(course_stats)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to insert course stats: {str(e)}")
            raise

    def insert_shot_data(self, shot_data, collection_name= "shot_data"):
        try:
            collection = self.db[collection_name]
            
            # Basic validation
            required_fields = ['tournament_id', 'course_id', 'holes_data', 'player_stats']
            if not all(field in shot_data for field in required_fields):
                raise ValueError("Missing required fields in shot data")
                
            result = collection.insert_one(shot_data)
            logger.info("Successfully inserted shot data")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to insert shot data: {str(e)}")
            raise
        
    def insert_coursefit_stats(self, field_stats, collection_name = "field_stats"):
        try:
            collection = self.db[collection_name]
            collection.update_one(
                {
                    "tournament_id": field_stats["tournament_id"],
                    "field_stat_type": field_stats["field_stat_type"]
                },
                {"$set": field_stats},
                upsert=True
            )
            self.logger.info("Successfully stored course fit stats")
        except Exception as e:
            self.logger.error(f"Failed to insert course fit stats: {str(e)}")
            raise
        
        

    def insert_tournament_history(self, history_data, collection_name = "tournament_history"):
        try:
            collection = self.db[collection_name]  
            
            # Create index if it doesn't exist
            if 'tournament_id_1_year_1' not in collection.index_information():
                collection.create_index([
                    ("tournament_id", 1),
                    ("year", 1)
                ], unique=True)
            
            # Handle single dict or list
            if isinstance(history_data, dict):
                history_data = [history_data]
                
            inserted_count = 0
            updated_count = 0
            duplicate_count = 0
            total_count = len(history_data)
            
            for year_data in history_data:
                try:
                    if "tournament_id" in year_data and year_data["tournament_id"].startswith("R"):
                        base_id = year_data["tournament_id"][5:] 
                        current_year = datetime.now().year
                        year_data["original_tournament_id"] = year_data["tournament_id"]
                        year_data["tournament_id"] = f"R{current_year}{base_id}"
        
                    result = collection.update_one(
                        {
                            "tournament_id": year_data["tournament_id"],
                            "year": year_data["year"]
                        },
                        {"$set": year_data},
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        inserted_count += 1
                        self.logger.debug(f"Inserted new record for year {year_data.get('year')}")
                    elif result.modified_count > 0:
                        updated_count += 1
                        self.logger.debug(f"Updated existing record for year {year_data.get('year')}")
                    else:
                        duplicate_count += 1
                        self.logger.debug(f"Skipped duplicate record for year {year_data.get('year')}")
                        
                except Exception as e:
                    self.logger.error(f"Error inserting year {year_data.get('year')}: {str(e)}")
                    continue
                    
            self.logger.info(f"Tournament history processing complete. Total: {total_count}, "
                            f"Inserted: {inserted_count}, Updated: {updated_count}, "
                            f"Duplicates/Unchanged: {duplicate_count}")
        except Exception as e:
            self.logger.error(f"Failed to insert tournament history: {str(e)}")
            raise
            
    def insert_current_form(self, current_form_data, collection_name = "current_form"):
        try:
            collection = self.db[collection_name]
            
            collection.create_index([
                ("tournament_id", 1),
                ("collected_at", -1)
            ])
            
            collection.update_one(
                {"tournament_id": current_form_data["tournament_id"]},
                {"$set": current_form_data},
                upsert=True
            )
            
            self.logger.info("Successfully stored current form stats")
        except Exception as e:
            self.logger.error(f"Failed to insert current form stats: {str(e)}")
            raise
        
        
        
    def insert_tournament_history_stats(self, history_stats, collection_name = "tournament_history_stats"):
        try:
            collection = self.db[collection_name]
            
            # Create indexes
            collection.create_index([
                ("tournament_id", 1),
                ("collected_at", -1)
            ])
            
            # Update or insert the document
            collection.update_one(
                {"tournament_id": history_stats["tournament_id"]},
                {"$set": history_stats},
                upsert=True
            )
            
            self.logger.info("Successfully stored tournament history stats")
        except Exception as e:
            self.logger.error(f"Failed to insert tournament history stats: {str(e)}")
            raise


    def get_player_round_stats(self, tournament_id, round_num = None):
        try:
            collection = self.db["shot_data"]
            query = {"tournament_id": tournament_id}
            
            if round_num is not None:
                query["holes_data.rounds.round_number"] = round_num
                
            return collection.find_one(query)
        except Exception as e:
            logger.error(f"Failed to retrieve round stats: {str(e)}")
            return None

    def get_hole_stats(self, tournament_id, course_id, hole_number):
        try:
            collection = self.db["shot_data"]
            return collection.find_one({
                "tournament_id": tournament_id,
                "course_id": course_id,
                "holes_data.hole_number": hole_number
            })
        except Exception as e:
            logger.error(f"Failed to retrieve hole stats: {str(e)}")
            return None
        
        
    def get_tournament_history_by_year_range(self, tournament_id, start_year, end_year):
        try:
            collection = self.db["tournament_history"]
            result = list(collection.find({
                "tournament_id": tournament_id,
                "year": {"$gte": start_year, "$lte": end_year}
            }).sort("year", 1))
            
            self.logger.info(f"Retrieved {len(result)} historical records for tournament {tournament_id} "
                            f"from {start_year} to {end_year}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to retrieve tournament history: {str(e)}")
            return []
        
        
    def insert_scorecards(self, scorecard_data, collection_name = "scorecards"):

        try:
            collection = self.db[collection_name]
            
            if 'id_1_player_id_1' not in collection.index_information():
                collection.create_index([
                    ("id", 1),
                    ("player_id", 1)
                ])
            if 'collected_at_-1' not in collection.index_information():
                collection.create_index([("collected_at", -1)])
            

            if isinstance(scorecard_data, list):
                if not scorecard_data:  
                    self.logger.warning("No scorecard data to insert")
                    return None
                result = collection.insert_many(scorecard_data)
                self.logger.info(f"Successfully stored {len(result.inserted_ids)} scorecards")
                return result.inserted_ids
            elif isinstance(scorecard_data, dict):
                result = collection.insert_one(scorecard_data)
                self.logger.info(f"Successfully stored scorecard for player {scorecard_data.get('player_id')}")
                return result.inserted_id
            else:
                self.logger.warning(f"Invalid scorecard data type: {type(scorecard_data)}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to insert scorecards: {str(e)}")
            raise
        
        ##weather
        
    def insert_tournament_weather(self, weather_data, collection_name = "tournament_weather"):
        try:
            collection = self.db[collection_name]
            
            if 'tournament_id_1_year_1' not in collection.index_information():
                collection.create_index([
                    ("tournament_id", 1),
                    ("year", 1)
                ], unique=True)
            
            result = collection.update_one(
                {
                    "tournament_id": weather_data["tournament_id"],
                    "year": weather_data["year"]
                },
                {"$set": weather_data},
                upsert=True
            )
            
            if result.upserted_id:
                self.logger.info(f"Inserted new weather data for tournament {weather_data['tournament_id']}")
            elif result.modified_count > 0:
                self.logger.info(f"Updated existing weather data for tournament {weather_data['tournament_id']}")
            else:
                self.logger.info(f"No changes to weather data for tournament {weather_data['tournament_id']}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to insert tournament weather data: {str(e)}")
            raise
            
    def get_tournament_weather(self, tournament_id, year):
        try:
            collection = self.db["tournament_weather"]
            query = {"tournament_id": tournament_id}
            
            if year is not None:
                query["year"] = year
                
            result = collection.find_one(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve tournament weather: {str(e)}")
            return None
        
        
        
        #players career and overview

    # Add these methods to your DatabaseManager class

    def insert_player_career(self, career_data, collection_name = "player_career"):
        try:
            collection = self.db[collection_name]
            
            # Create indexes if they don't exist
            if 'player_id_1_tour_code_1' not in collection.index_information():
                collection.create_index([
                    ("player_id", 1),
                    ("tour_code", 1)
                ], unique=True)
            
            # Update or insert the document
            result = collection.update_one(
                {
                    "player_id": career_data["player_id"],
                    "tour_code": career_data["tour_code"]
                },
                {"$set": career_data},
                upsert=True
            )
            
            if result.upserted_id:
                self.logger.info(f"Inserted new career data for player {career_data['player_id']}")
            elif result.modified_count > 0:
                self.logger.info(f"Updated existing career data for player {career_data['player_id']}")
            else:
                self.logger.info(f"No changes to career data for player {career_data['player_id']}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to insert player career data: {str(e)}")
            raise

    def insert_player_profile_overview(self, overview_data, collection_name = "player_profile_overview"):
        
        try:
            collection = self.db[collection_name]
            
            if "standings" in overview_data and overview_data["standings"] is not None:
                standings = overview_data["standings"]
                if "owgr" in standings and standings["owgr"] is not None:
                    self.logger.info(f"Player {overview_data['player_id']} has OWGR: {standings['owgr']}")
                    
                    owgr_value = standings["owgr"]
                    if owgr_value is not None:
                        overview_data["standings"]["owgr"] = owgr_value
                        overview_data["debug_owgr"] = owgr_value
                else:
                    self.logger.warning(f"Player {overview_data['player_id']} missing OWGR value")
            result = collection.update_one(
                {"player_id": overview_data["player_id"]},
                {"$set": overview_data},
                upsert=True
            )
            
            if result.upserted_id:
                self.logger.info(f"Inserted new profile overview for player {overview_data['player_id']}")
            elif result.modified_count > 0:
                self.logger.info(f"Updated existing profile overview for player {overview_data['player_id']}")
            else:
                self.logger.info(f"No changes to profile overview for player {overview_data['player_id']}")
            
            stored_doc = collection.find_one({"player_id": overview_data["player_id"]})
            if stored_doc and "standings" in stored_doc and stored_doc["standings"] is not None:
                if "owgr" in stored_doc["standings"] and stored_doc["standings"]["owgr"] is not None:
                    self.logger.info(f"Verified OWGR in database for player {overview_data['player_id']}: {stored_doc['standings']['owgr']}")
                else:
                    self.logger.warning(f"OWGR missing in database for player {overview_data['player_id']} after insert!")
                    
                    if "standings" in overview_data and overview_data["standings"] is not None and "owgr" in overview_data["standings"]:
                        owgr = overview_data["standings"]["owgr"]
                        collection.update_one(
                            {"player_id": overview_data["player_id"]},
                            {"$set": {"standings.owgr": owgr}}
                        )
                        self.logger.info(f"Attempted direct OWGR update for player {overview_data['player_id']}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to insert player profile overview: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def get_player_career(self, player_id, tour_code = "R"):
        try:
            collection = self.db["player_career"]
            result = collection.find_one({
                "player_id": player_id,
                "tour_code": tour_code
            })
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve player career data: {str(e)}")
            return None

    def get_player_profile_overview(self, player_id):
        try:
            collection = self.db["player_profile_overview"]
            return collection.find_one({"player_id": player_id})
        except Exception as e:
            self.logger.error(f"Failed to get player profile: {str(e)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve player profile overview: {str(e)}")
            return None
        
        
        
        
    def run_query(self, collection_name, query, projection = None):
        try:
            collection = self.db[collection_name]
            if projection:
                results = list(collection.find(query, projection))
            else:
                results = list(collection.find(query))
            self.logger.info(f"Retrieved {len(results)} documents from {collection_name}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to run query on {collection_name}: {str(e)}")
            raise
        
        
        
    def collection_exists(self, collection_name):
        try:
            return collection_name in self.db.list_collection_names()
        except Exception as e:
            self.logger.error(f"Error checking if collection {collection_name} exists: {str(e)}")
            return False
        
        
    def get_collection_stats(self, collection_name):
        try:
            if not self.collection_exists(collection_name):
                return {"exists": False, "count": 0}
            
            count = self.db[collection_name].count_documents({})
            return {
                "exists": True,
                "count": count,
                "indexes": list(self.db[collection_name].index_information().keys())
            }
        except Exception as e:
            self.logger.error(f"Error getting stats for collection {collection_name}: {str(e)}")
            return {"error": str(e)}    
        
        
        
        
        
    def get_players_batch(self, player_ids, projection = None):
        try:
            if not player_ids:
                return {}

            collection = self.db["player_profile_overview"]
            query = {"player_id": {"$in": player_ids}}
        
            if projection is None:
                projection = {
                    "_id": 0,
                    "player_id": 1,
                    "name": 1,
                    "owgr": 1,
                    "country": 1,
                    "standings": 1
                }
            
            cursor = collection.find(query, projection)
            return {doc["player_id"]: doc for doc in cursor}
            
        except Exception as e:
            self.logger.error(f"Batch player fetch failed: {str(e)}")
            return {}

    def get_tournament_players_batch(self, tournament_id, year, projection = None) :
        try:
            collection = self.db["tournament_history"]
            query = {"tournament_id": tournament_id, "year": year}
            
            # Default projection if not specified
            if projection is None:
                projection = {
                    "_id": 0,
                    "players.player_id": 1,
                    "players.name": 1,
                    "players.position": 1,
                    "players.score": 1
                }
            
            result = collection.find_one(query, projection)
            return result.get("players", []) if result else []
            
        except Exception as e:
            self.logger.error(f"Tournament players batch fetch failed: {str(e)}")
            return []
        
        
        
    def get_tournament_history_optimized(self, tournament_id: str, player_ids, year):
        try:
            collection = self.db["tournament_history"]
            
            pipeline = [
                {"$match": {
                    "tournament_id": tournament_id,
                    "year": year,
                    "players.player_id": {"$in": player_ids}
                }},
                {"$unwind": "$players"},
                {"$match": {
                    "players.player_id": {"$in": player_ids}
                }},
                {"$group": {
                    "_id": "$players.player_id",
                    "data": {"$first": "$players"}
                }}
            ]
            
            results = {}
            for doc in collection.aggregate(pipeline, allowDiskUse=True):
                results[doc["_id"]] = doc["data"]
                
            return results
            
        except Exception as e:
            self.logger.error(f"Optimized history fetch failed: {str(e)}")
            return {}