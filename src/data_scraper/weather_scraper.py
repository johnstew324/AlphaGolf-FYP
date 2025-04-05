import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import asyncio
import time

class TournamentWeather(BaseModel):
    tournament_id: str
    tournament_name: str
    course_name: str
    year: int
    location: str
    days: List[Dict]
    collected_at: datetime = Field(default_factory=datetime.utcnow)

class WeatherScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.min_request_interval = 1.0  # Minimum interval between requests in seconds
        self.last_request_time = 0

    async def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()

    def parse_date_range(self, date_string: str, year: int) -> tuple:
        if not date_string:
            return None, None
        
        year_str = str(year)
        
        range_parts = date_string.split(" - ")
        if len(range_parts) != 2:
            self.logger.warning(f"Invalid date format: {date_string}")
            return None, None
        
        start_part = range_parts[0].strip()
        end_part = range_parts[1].strip()

        start_parts = start_part.split()
        if len(start_parts) < 2:
            self.logger.warning(f"Invalid start date format: {start_part}")
            return None, None
        
        start_month = start_parts[0]
        start_day = start_parts[1]
        
        end_parts = end_part.split()
        if len(end_parts) == 1:
            end_month = start_month
            end_day = end_parts[0]
        else:
            end_month = end_parts[0]
            end_day = end_parts[1]
        
        start_date = f"{start_month} {start_day}, {year_str}"
        end_date = f"{end_month} {end_day}, {year_str}"
        
        try:
            start = datetime.strptime(start_date, "%b %d, %Y")
            end = datetime.strptime(end_date, "%b %d, %Y")
            
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        except ValueError as e:
            self.logger.error(f"Error parsing dates: {date_string} - {e}")
            return None, None

    async def fetch_weather(self, location: str, start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch weather data from Visual Crossing API"""
        await self._wait_for_rate_limit()
        
        # URL encode the location
        encoded_location = requests.utils.quote(location)
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{encoded_location}/{start_date}/{end_date}"
        
        params = {
            'unitGroup': 'us',  # Units: us (fahrenheit) or metric
            'key': self.api_key,
            'include': 'days',
            'contentType': 'json',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,dew,precip,precipprob,precipcover,windgust,windspeed,winddir,pressure,cloudcover,solarradiation,solarenergy,uvindex,sunrise,sunset,snowdepth,conditions'
        }
        
        try:
            response = requests.get(url, params=params)
            log_url = response.url.replace(self.api_key, 'API_KEY_HIDDEN')
            self.logger.info(f"API URL: {log_url}")
            
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP Error fetching weather for {location}: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                self.logger.error(f"Response text: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching weather for {location}: {e}")
            return None

    def get_tournament_location(self, tournament: Dict) -> str:
        city = tournament.get("City", "")
        state = tournament.get("State", "")
        country = tournament.get("Country", "USA")
        
        if city and state and country == "United States of America":
            return f"{city}, {state}, USA"
        elif city and country:
            return f"{city}, {country}"
        else:
            return tournament.get("Location", city).strip(", ")

    async def scrape_tournament_weather(self, tournament_data: Dict, year: int) -> Optional[Dict]:
        try:
            tournament_id = tournament_data.get("Tournament ID", "Unknown")
            tournament_name = tournament_data.get("Tournament Name", "Unknown")
            date_str = tournament_data.get("Date", "")
            course_name = tournament_data.get("Course Name", "Unknown")
            
            location = self.get_tournament_location(tournament_data)
            
            start_date, end_date = self.parse_date_range(date_str, year)
            
            self.logger.info(f"Processing: {tournament_name} ({date_str}) at {location}")
            
            if not start_date or not end_date:
                self.logger.warning(f"Could not parse dates for {tournament_name}, skipping...")
                return None
            
            self.logger.info(f"Fetching weather for {tournament_name} ({start_date} to {end_date}) at {location}")
            weather_data = await self.fetch_weather(location, start_date, end_date)
            
            if not weather_data:
                self.logger.warning(f"No weather data found for {tournament_name}")
                return None
                
            days = weather_data.get("days", [])
            if not days:
                self.logger.warning(f"No days/rounds data found for {tournament_name}")
                return None
            
            tournament_weather = {
                "tournament_id": tournament_id,
                "tournament_name": tournament_name,
                "course_name": course_name,
                "year": year,
                "location": location,
                "rounds": days,# Changed from days to rounds
                "collected_at": datetime.utcnow()
            }
            return tournament_weather
        

        except Exception as e:
            self.logger.error(f"Error scraping tournament weather: {str(e)}")
            return None

    async def scrape_from_json(self, json_file: str) -> List[Dict]:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Loaded JSON file: {json_file}")
            
            tournaments = []
            results = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "Year" in item:
                        year = item.get("Year")
                        self.logger.info(f"Processing year: {year}")
                        
                        year_tournaments = []
                        if "Completed Tournaments" in item:
                            year_tournaments.extend(item.get("Completed Tournaments", []))
                        if "Upcoming Tournaments" in item:
                            year_tournaments.extend(item.get("Upcoming Tournaments", []))
                            
                        for tournament in year_tournaments:
                            weather_data = await self.scrape_tournament_weather(tournament, year)
                            if weather_data:
                                results.append(weather_data)
                            await asyncio.sleep(0.5)
            
            self.logger.info(f"Scraped weather data for {len(results)} tournaments")
            return results
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in file: {json_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading tournament data: {str(e)}")
            return []

    async def scrape_tournament_weather_by_id(self, tournament_id: str, year: int, tournament_data: Dict) -> Optional[Dict]:
        try:
            if tournament_data.get("Tournament ID") == tournament_id:
                return await self.scrape_tournament_weather(tournament_data, year)
            return None
        except Exception as e:
            self.logger.error(f"Error scraping tournament weather by ID: {str(e)}")
            return None