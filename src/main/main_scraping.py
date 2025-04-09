import asyncio
import logging
from datetime import datetime
from config import Config
from database import DatabaseManager
from data_scraper.player_stats_scraper import PlayerStatsScraper
from data_scraper.course_stats_scraper import CourseStatsScraper
from data_scraper.shot_stats_scraper import ShotDataScraper
from data_scraper.tournament_history_scraper import TournamentHistoryScraper
from data_scraper.course_fit_stats_scraper import CourseFitStatsScraper
from data_scraper.scorecard_scraper import ScorecardScraper
from data_scraper.current_form_scraper import CurrentFormScraper
from data_scraper.tournament_history_stats_scraper import TournamentHistoryStatsScraper
from data_scraper.weather_scraper import WeatherScraper
from data_scraper.player_career_scraper import PlayerCareerScraper
from data_scraper.player_overview_scraper import PlayerProfileOverviewScraper
import time
import os
import json


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set the years for tournament scraping and player stats
TOURNAMENT_YEARS = [2014, 2015,2016,2017,2018,2019, 2020, 2021, 2022, 2023, 2024, 2025]
PLAYER_STATS_YEARS = [2014, 2015,2016,2017,2018,2019, 2020, 2021, 2022, 2023, 2024, 2025]
HISTORY_START_YEAR = 2007  # Start historical data from 2007
HISTORY_END_YEAR = 2025    

# File paths for JSON data
PLAYERS_JSON = "C:\\Users\\johns\\AlphaGOLF-FYP\\AlphaGolf\\data\\raw_jsons\\players.json"
TOURNAMENTS_JSON= "C:\\Users\\johns\\AlphaGOLF-FYP\\AlphaGolf\\data\\raw_jsons\\tournaments.json"
TOURNAMENTS_WEATHER_JSON = "C:\\Users\\johns\\AlphaGOLF-FYP\\AlphaGolf\\data\\raw_jsons\\tournaments_location_dates.json"

# Load JSON data
def load_json_data():
    try:
        # Load players data
        with open(PLAYERS_JSON, 'r') as f:
            players_data = json.load(f)
        logger.info(f"Loaded {len(players_data)} players from JSON file")
        
        # Filter to only active players
        active_players = [player for player in players_data if player.get('isActive', False)]
        logger.info(f"Found {len(active_players)} active players")
        
        # Load tournaments data
        with open(TOURNAMENTS_JSON, 'r') as f:
            tournaments_data = json.load(f)
        logger.info(f"Loaded {len(tournaments_data)} tournaments from JSON file")
        
        # Load tournament weather data
        with open(TOURNAMENTS_WEATHER_JSON, 'r') as f:
            tournaments_weather_data = json.load(f)
        logger.info(f"Loaded tournament weather data from JSON file")
        
        return active_players, tournaments_data, tournaments_weather_data
    except Exception as e:
        logger.error(f"Error loading JSON data: {str(e)}")
        return [], [], []

async def safe_scrape_and_store(scraper_name: str, scraper_func, db_store_func, *args, **kwargs):
    """Safely execute a scrape operation and store the results"""
    try:
        logger.info(f"Starting {scraper_name}")
        data = await scraper_func(*args, **kwargs)
        
        if data:
            # Handle potential None values in stat_headers for certain scrapers
            if isinstance(data, dict) and 'stat_headers' in data and data['stat_headers'] is None:
                data['stat_headers'] = []

            # Log some basic information about the data before storing
            if isinstance(data, dict):
                if 'tournament_id' in data:
                    logger.info(f"Data for tournament: {data.get('tournament_id')}")
                if 'course_id' in data:
                    logger.info(f"Data for course: {data.get('course_id')}")
                if 'rounds' in data:
                    logger.info(f"Number of rounds: {len(data.get('rounds', []))}")
                
            db_store_func(data)
            logger.info(f"Successfully stored {scraper_name} data")
            return True
        else:
            logger.warning(f"No {scraper_name} data available")
            return False
            
    except Exception as e:
        logger.error(f"Error in {scraper_name}: {str(e)}")
        return False

async def scrape_tournament_weather(weather_scraper: WeatherScraper, tournament_data: dict, db_manager: DatabaseManager):
    """Scrape weather data for a single tournament"""
    try:
        tournament_id = tournament_data.get("Tournament ID")
        tournament_name = tournament_data.get("Tournament Name")
        year = int(tournament_id[1:5]) if tournament_id.startswith("R") and len(tournament_id) >= 5 else 2025
        
        logger.info(f"Scraping weather for {tournament_name} (ID: {tournament_id}, Year: {year})")
        
        weather_data = await weather_scraper.scrape_tournament_weather(tournament_data, year)
        
        if weather_data:
            db_manager.insert_tournament_weather(weather_data)
            logger.info(f"Successfully stored weather data for {tournament_name}")
            return True
        else:
            logger.warning(f"No weather data available for {tournament_name}")
            return False
    
    except Exception as e:
        logger.error(f"Error scraping tournament weather: {str(e)}")
        return False

async def scrape_weather_from_json(db_manager: DatabaseManager, api_key: str, tournaments_weather_data: list):
    """Scrape weather data from provided tournament weather data"""
    try:
        if not api_key:
            logger.error("Weather API key not found in configuration")
            return False
        
        # Extract the tournaments data from the structure
        all_tournaments = []
        for year_data in tournaments_weather_data:
            if isinstance(year_data, dict) and "Year" in year_data:
                year = year_data.get("Year")
                logger.info(f"Processing weather data for year: {year}")
                
                # Get tournaments for this year
                completed = year_data.get("Completed Tournaments", [])
                upcoming = year_data.get("Upcoming Tournaments", [])
                all_tournaments.extend(completed + upcoming)
        
        logger.info(f"Initializing weather scraper for {len(all_tournaments)} tournaments")
        weather_scraper = WeatherScraper(api_key)
        
        # Process each tournament
        processed_count = 0
        for tournament in all_tournaments:
            try:
                if await scrape_tournament_weather(weather_scraper, tournament, db_manager):
                    processed_count += 1
                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing tournament weather: {str(e)}")
                continue
                
        logger.info(f"Successfully processed weather data for {processed_count} tournaments")
        return True
        
    except Exception as e:
        logger.error(f"Error in weather scraping from JSON: {str(e)}")
        return False

async def scrape_tournament_data(tournament: dict, db_manager: DatabaseManager, scrapers: dict):
    """Scrape all data for a single tournament with proper handling of round data"""
    try:
        tournament_id = tournament.get("id")
        if not tournament_id.startswith("R") and len(tournament_id) <= 3:
            # Handle base tournament IDs - format for each year
            base_id = tournament_id
            tournament_ids = []
            for year in TOURNAMENT_YEARS:
                tournament_ids.append(f"R{year}{base_id}")
            
            logger.info(f"Processing tournament: {tournament.get('name')} with base ID {base_id}")
            logger.info(f"Generated {len(tournament_ids)} tournament IDs for different years")
            
            # Process each year's tournament ID
            for tournament_id in tournament_ids:
                await scrape_single_tournament_id(tournament_id, tournament.get('name', 'Unknown'), db_manager, scrapers)
                # Add delay between tournament years to respect rate limits
                await asyncio.sleep(2)
        else:
            # Already a formatted tournament ID
            await scrape_single_tournament_id(tournament_id, tournament.get('name', 'Unknown'), db_manager, scrapers)
            
    except Exception as e:
        logger.error(f"Error processing tournament {tournament.get('name', 'unknown')}: {str(e)}")

async def scrape_single_tournament_id(tournament_id: str, tournament_name: str, db_manager: DatabaseManager, scrapers: dict):
    """Scrape data for a single tournament ID"""
    try:
        logger.info(f"Processing tournament ID: {tournament_id}")
        
        # Extract year from tournament_id (assuming format like "R2024007")
        tournament_year = None
        if tournament_id.startswith("R") and len(tournament_id) >= 5:
            try:
                tournament_year = int(tournament_id[1:5])
            except ValueError:
                tournament_year = None
                
        logger.info(f"Tournament year: {tournament_year}")
        
        # 1. Course data - available for all years
        courses = await scrapers['course'].get_tournament_courses(tournament_id)
        if courses:
            logger.info(f"Found {len(courses)} courses for tournament {tournament_id}")
            
            for course in courses:
                # Use test_scrape_course_stats which we know works correctly
                logger.info(f"Scraping course {course['course_name']} (ID: {course['course_id']})")
                course_data = await scrapers['course'].test_scrape_course_stats(tournament_id, course['course_id'])
                
                if course_data:
                    # Log round information to verify
                    rounds = course_data.get('rounds', [])
                    
                    # This is critical - check that we have rounds data
                    if rounds:
                        logger.info(f"Scraped {len(rounds)} rounds for course {course['course_name']}")
                        
                        # Log each round
                        for i, round_data in enumerate(rounds):
                            round_num = round_data.get('round_number')
                            round_header = round_data.get('round_header')
                            hole_count = len(round_data.get('holes', []))
                            logger.info(f"Round {round_num} ({round_header}): {hole_count} holes")
                        
                        # Store in database
                        db_manager.insert_course_stats(course_data)
                        logger.info(f"Stored course stats with {len(rounds)} rounds for {course['course_name']}")
                    else:
                        logger.warning(f"No rounds data found for course {course['course_name']}")
                else:
                    logger.warning(f"No course stats available for {course['course_name']}")
                
                # For the following data types, only scrape if tournament is from 2024 or 2025
                if tournament_year and tournament_year >= 2024:
                    logger.info(f"Tournament {tournament_id} is from {tournament_year}, scraping shot data...")
                    
                    # Shot data - only from 2024 onwards
                    await safe_scrape_and_store(
                        "shot data",
                        scrapers['shot'].scrape_shot_data,
                        db_manager.insert_shot_data,
                        tournament_id,
                        int(course['course_id'])
                    )
                else:
                    logger.info(f"Tournament {tournament_id} is before 2024, skipping shot data")
        
        # For the following data types, only scrape if tournament is from 2024 or 2025
        if tournament_year and tournament_year >= 2024:
            logger.info(f"Tournament {tournament_id} is from {tournament_year}, scraping recent stats...")
            
            # 2. Course fit stats
            await safe_scrape_and_store(
                "course fit stats",
                scrapers['coursefit'].scrape_coursefit_stats,
                db_manager.insert_coursefit_stats,
                tournament_id
            )
            
            # 3. Current form stats
            await safe_scrape_and_store(
                "current form stats",
                scrapers['current_form'].scrape_current_form,
                db_manager.insert_current_form,
                tournament_id
            )
            
            # 4. Tournament history stats
            await safe_scrape_and_store(
                "tournament history stats",
                scrapers['history_stats'].scrape_tournament_history_stats,
                db_manager.insert_tournament_history_stats,
                tournament_id
            )
        else:
            logger.info(f"Tournament {tournament_id} is before 2024, skipping course fit, current form, and tournament history stats")
            
        logger.info(f"Completed processing tournament ID: {tournament_id}")
        
    except Exception as e:
        logger.error(f"Error processing tournament ID {tournament_id}: {str(e)}")

async def scrape_historical_data(tournament: dict, db_manager: DatabaseManager, scrapers: dict):
    """
    Scrape historical tournament data for a single tournament
    """
    try:
        tournament_id = tournament.get("id")
        tournament_name = tournament.get("name", "Unknown")
        
        # Format the tournament ID with the most recent year for historical data
        if not tournament_id.startswith("R"):
            most_recent_tournament_id = f"R{max(TOURNAMENT_YEARS)}{tournament_id}"
        else:
            most_recent_tournament_id = tournament_id
            
        logger.info(f"Scraping historical data for {tournament_name} (years {HISTORY_START_YEAR}-{HISTORY_END_YEAR}) using ID: {most_recent_tournament_id}")
        
        # Get all historical data in one call
        historical_data = await scrapers['history'].scrape_historical_data(
            most_recent_tournament_id,
            HISTORY_START_YEAR,  # Much broader range
            HISTORY_END_YEAR
        )
        
        if historical_data:
            logger.info(f"Retrieved {len(historical_data)} years of historical data for {tournament_name}")
            db_manager.insert_tournament_history(historical_data)
            logger.info(f"Successfully stored historical tournament data for {tournament_name}")
            return True
        else:
            logger.warning(f"No historical tournament data retrieved for {tournament_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error scraping historical data for {tournament_name}: {str(e)}")
        return False

async def scrape_player_data(player: dict, all_tournaments: list, db_manager: DatabaseManager, scrapers: dict):
    """Scrape all data for a single player"""
    try:
        player_id = player.get('id')
        player_name = player.get('name')
        
        if not player_id or not player_name:
            logger.warning(f"Missing player ID or name in player data: {player}")
            return
            
        logger.info(f"Processing player: {player_name} (ID: {player_id})")
        
        # 1. Player stats for recent years
        for year in PLAYER_STATS_YEARS:
            await safe_scrape_and_store(
                f"player stats for {year}",
                scrapers['player'].scrape_player_stats,
                db_manager.insert_player_stats,
                player_id,
                player_name,
                year
            )
            
        # 2. Player career data
        await safe_scrape_and_store(
            "player career data",
            scrapers['player_career'].scrape_player_career,
            db_manager.insert_player_career,
            player_id,
            "R"  # PGA Tour code
        )
        
        # 3. Player profile overview data
        await safe_scrape_and_store(
            "player profile overview",
            scrapers['player_overview'].scrape_player_profile_overview,
            db_manager.insert_player_profile_overview,
            player_id,
            "R"  # PGA Tour code
        )
            
        # 4. Get scorecards for recent tournaments (2024-2025 only)
        recent_tournament_ids = []
        for tournament in all_tournaments:
            tournament_id = tournament.get("id")
            if not tournament_id.startswith("R"):
                # Generate IDs for recent years only
                for year in [2024, 2025]:
                    recent_tournament_ids.append(f"R{year}{tournament_id}")
            elif tournament_id.startswith("R") and len(tournament_id) >= 5:
                try:
                    year = int(tournament_id[1:5])
                    if year >= 2024:
                        recent_tournament_ids.append(tournament_id)
                except ValueError:
                    continue
        
        logger.info(f"Getting scorecards for {len(recent_tournament_ids)} recent tournaments")
        for tournament_id in recent_tournament_ids:
            # Get scorecard directly, not through scrape_multiple_scorecards
            scorecard = await scrapers['scorecard'].scrape_scorecard(tournament_id, player_id)
            if scorecard:
                try:
                    db_manager.insert_scorecards(scorecard)
                    logger.info(f"Successfully stored scorecard for tournament {tournament_id}")
                except Exception as e:
                    logger.error(f"Error storing scorecard for {tournament_id}: {str(e)}")
            else:
                logger.warning(f"No scorecard available for tournament {tournament_id}")
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        logger.info(f"Completed processing player: {player_name}")
        
    except Exception as e:
        logger.error(f"Error processing player {player.get('name', 'unknown')}: {str(e)}")

async def main_scraper():
    """Main function for scraping data from JSON files"""
    try:
        # Load data from JSON files
        active_players, tournaments_data, tournaments_weather_data = load_json_data()
        
        if not active_players or not tournaments_data:
            logger.error("Failed to load required data from JSON files")
            return
            
        # Initialize database
        db_manager = DatabaseManager(Config.MONGODB_URI)
        logger.info("Database connection established")
        
        # Initialize all scrapers
        scrapers = {
            'course': CourseStatsScraper(Config.BASE_URL, Config.HEADERS),
            'shot': ShotDataScraper(Config.BASE_URL, Config.HEADERS),
            'coursefit': CourseFitStatsScraper(Config.BASE_URL, Config.HEADERS),
            'current_form': CurrentFormScraper(Config.BASE_URL, Config.HEADERS),
            'history_stats': TournamentHistoryStatsScraper(Config.BASE_URL, Config.HEADERS),
            'history': TournamentHistoryScraper(Config.BASE_URL, Config.HEADERS),
            'player': PlayerStatsScraper(Config.BASE_URL, Config.HEADERS),
            'scorecard': ScorecardScraper(Config.BASE_URL, Config.HEADERS),
            'player_career': PlayerCareerScraper(Config.BASE_URL, Config.HEADERS),
            'player_overview': PlayerProfileOverviewScraper(Config.BASE_URL, Config.HEADERS)
        }
        
        logger.info(f"Processing {len(tournaments_data)} tournaments and {len(active_players)} active players")
        
        # 0. Weather data scraping with tournament weather data
        if Config.VISUAL_CROSSING_API_KEY and tournaments_weather_data:
            logger.info("Starting weather data scraping...")
            await scrape_weather_from_json(db_manager, Config.VISUAL_CROSSING_API_KEY, tournaments_weather_data)
        else:
            logger.warning("Visual Crossing API key not found or no weather data available, skipping weather data scraping")
        
        # 1. Scrape historical data for all tournaments
        for tournament in tournaments_data:
            await scrape_historical_data(tournament, db_manager, scrapers)
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # 2. Process all tournament data
        for tournament in tournaments_data:
            await scrape_tournament_data(tournament, db_manager, scrapers)
            # Add delay between tournaments to respect rate limits
            await asyncio.sleep(2)
        
        # 3. Process all active player data
        for player in active_players:
            await scrape_player_data(player, tournaments_data, db_manager, scrapers)
            # Add delay between players to respect rate limits
            await asyncio.sleep(1)
        
        logger.info("Focused scraping completed successfully!")
        
    except Exception as e:
        logger.error(f"Focused scraper failed: {str(e)}")
        
    finally:
        # Clean up all scrapers
        for scraper in scrapers.values():
            await scraper.close()

async def main():
    """Entry point"""
    start_time = time.time()
    logger.info("Starting main scraping for all tournaments and active players from JSON files...")
    
    try:
        await main_scraper()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Scraping process completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    asyncio.run(main())