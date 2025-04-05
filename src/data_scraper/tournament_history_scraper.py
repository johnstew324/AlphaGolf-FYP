from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
from pydantic import BaseModel, Field, validator
import aiohttp

class PlayerRound(BaseModel):
    score: int = 0
    par_relative: int = 0
    
    class Config:
        extra = "allow"
    
    @validator('par_relative', 'score', pre=True)
    def handle_invalid_values(cls, v):
        if v is None or v == '':
            return 0
        if v == '-':
            return 0
        if v == 'E':  # 'E' means Even par which is 0
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0  # Default to 0 if can't convert

class PlayerResult(BaseModel):
    player_id: str
    name: str
    position: str
    country: Optional[str]
    total_score: int = 0
    par_relative: int = 0
    rounds: List[PlayerRound]
    
    @validator('total_score', 'par_relative', pre=True)
    def validate_score_fields(cls, v):
        if v is None or v == '' or v == '-':
            return 0
        if v == 'E':  
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0  

class TournamentWinner(BaseModel):
    player_id: Optional[str]
    name: str
    total_strokes: Optional[int]
    total_score: Optional[int]
    country: Optional[str]

class TournamentHistory(BaseModel):
    tournament_id: str
    year: int
    players: List[PlayerResult]
    winner: TournamentWinner
    collected_at: datetime = Field(default_factory=datetime.utcnow)

class TournamentHistoryScraper:
    def __init__(self, url: str, headers: Dict[str, str]):
        self.url = url
        self.headers = headers
        self.logger = logging.getLogger(__name__)
        self.query = """
        query TournamentHistory($id: ID!, $year: Int!) {
            tournamentPastResults(id: $id, year: $year) {
                id
                availableSeasons {
                    year
                    displaySeason
                }
                players {
                    id
                    position
                    player {
                        id
                        displayName
                        country
                        countryFlag
                    }
                    rounds {
                        score
                        parRelativeScore
                    }
                    additionalData
                    total
                    parRelativeScore
                }
                winner {
                    id
                    firstName
                    lastName
                    totalStrokes
                    totalScore
                    countryName
                }
            }
        }
        """
    async def close(self):
        try:
            if hasattr(self, 'session') and self.session and not self.session.closed:
                await self.session.close()
                self.logger.info("Session closed")
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")

# make graphql request to API
    async def make_request(self, query: str, variables: Dict) -> Optional[Dict]:
    
        try:
            payload = {
                "query": query,
                "variables": variables
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Request failed with status {response.status}")
                        return None
                    
                    return await response.json()
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}")
            return None

# parse response data
    async def parse_response(self, response_data: Dict, tournament_id: str, year: int) -> Optional[Dict]:
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id} in {year}")
                return None

            tournament_data = response_data.get('data', {}).get('tournamentPastResults')
            if not tournament_data:
                self.logger.warning(f"No tournament results found for {tournament_id} in {year}")
                return None

            player_results = []
            for player in tournament_data.get('players', []):
                try:
                    rounds = [
                        PlayerRound(
                            score=round.get('score', 0),
                            par_relative=round.get('parRelativeScore', 0)
                        )
                        for round in player.get('rounds', [])
                    ]

                    player_result = PlayerResult(
                        player_id=player['player']['id'],
                        name=player['player']['displayName'],
                        position=player['position'],
                        country=player['player'].get('country'),
                        total_score=player.get('total', 0),
                        par_relative=player.get('parRelativeScore', 0),
                        rounds=rounds
                    )
                    player_results.append(player_result)

                except Exception as e:
                    self.logger.error(f"Error processing player result: {str(e)}")
                    continue

            winner = tournament_data.get('winner', {})
            winner_data = TournamentWinner(
                player_id=winner.get('id'),
                name=f"{winner.get('firstName', '')} {winner.get('lastName', '')}".strip(),
                total_strokes=winner.get('totalStrokes'),
                total_score=winner.get('totalScore'),
                country=winner.get('countryName')
            )

            tournament_history = TournamentHistory(
                tournament_id=tournament_id,
                year=year,
                players=player_results,
                winner=winner_data
            )

            return tournament_history.dict()

        except Exception as e:
            self.logger.error(f"Error parsing tournament history: {str(e)}")
            return None

# scrape data for a specific year
    async def scrape_year_data(self, tournament_id: str, year: int) -> Optional[Dict]:

        try:
            
            year_code = year
            if year >= 2007:
                year_code = int(f"{year}0")  # Convert to format like 20070
            
            self.logger.info(f"Fetching data for tournament {tournament_id}, year {year} using year code {year_code}")
            
            variables = {
                "id": tournament_id,
                "year": year_code
            }

            response_data = await self.make_request(self.query, variables)
            
            if not response_data:
                self.logger.warning(f"No response data for tournament {tournament_id} in {year}")
                return None
            result = await self.parse_response(response_data, tournament_id, year)
            
            if result:

                self.logger.info(f"Successfully processed data for {tournament_id}, year {year}")
            
            return result

        except Exception as e:
            self.logger.error(f"Error scraping year data: {str(e)}")
            return None

# scrape historical using 2000 as a starting but this is changed in the main 
    async def scrape_historical_data(self, tournament_id: str, start_year: int = 2000,end_year: int = 2024) -> List[Dict]:
        try:
            self.logger.info(f"Fetching historical data for tournament {tournament_id} "
                         f"from {start_year} to {end_year}")
            processed_years = set() 
            valid_results = []
            
    
            for year in range(start_year, end_year + 1):
                result = await self.scrape_year_data(tournament_id, year)
                
                if result is not None:
                    year_value = result.get('year')
                    if year_value not in processed_years:
                        processed_years.add(year_value)
                        valid_results.append(result)
                        self.logger.info(f"Added year {year_value} for tournament {tournament_id}")
                    else:
                        self.logger.info(f"Skipping duplicate year {year_value} for tournament {tournament_id}")
                else:
                    self.logger.info(f"No data found for tournament {tournament_id} in year {year}")
                await asyncio.sleep(0.5)

            self.logger.info(f"Successfully collected data for {len(valid_results)} years")
            return valid_results

        except Exception as e:
            self.logger.error(f"Error scraping historical data: {str(e)}")
            return []

# scrape multiple tournaments
    async def scrape_multiple_tournaments(self,tournament_ids: List[str],start_year: int = 2000,end_year: int = 2024) -> Dict[str, List[Dict]]:
        try:
            self.logger.info(f"Processing {len(tournament_ids)} tournaments")
            
            tournament_data = {}
            
            # Process each tournament sequentially to avoid rate limiting
            for tournament_id in tournament_ids:
                self.logger.info(f"Scraping tournament {tournament_id}")
                data = await self.scrape_historical_data(tournament_id, start_year, end_year)
                tournament_data[tournament_id] = data
                
                # Add a delay between tournaments
                await asyncio.sleep(1)
            
            return tournament_data

        except Exception as e:
            self.logger.error(f"Error in processing multiple tournaments: {str(e)}")
            return {}