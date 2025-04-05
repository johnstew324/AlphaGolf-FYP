from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field, validator
import asyncio 


# Pydantic models for player profile overview data
# api query used 
class HeadshotData(BaseModel):
    country: str
    country_flag: Optional[str] = None
    first_name: str
    last_name: str

class StandingsData(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    total: Optional[Union[int, float, str]] = None
    total_label: Optional[str] = None
    rank: Optional[Union[int, str]] = None
    owgr: Optional[Union[int, str]] = None
    detail_copy: Optional[str] = None
    
    @validator('rank', 'owgr', pre=True)
    def validate_rank_fields(cls, v):
        if v is None or v == '':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return str(v)

class StatData(BaseModel):
    title: str
    value: Optional[str] = None
    career: Optional[str] = None
    wide: Optional[bool] = False

class PerformanceData(BaseModel):
    tour: str
    season: str
    display_season: str
    stats: List[StatData]

class SnapshotData(BaseModel):
    title: str
    value: str
    description: Optional[str] = None

class PlayerProfileOverviewData(BaseModel):
    player_id: str
    first_name: str
    last_name: str
    country: str
    standings: Optional[StandingsData] = None
    fedex_fall_standings: Optional[StandingsData] = None
    performance: List[PerformanceData] = []
    snapshot: List[SnapshotData] = []
    collected_at: datetime = Field(default_factory=datetime.utcnow)

class PlayerProfileOverviewScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query PlayerProfileOverview($playerId: ID!, $currentTour: TourCode) {
          playerProfileOverview(playerId: $playerId, currentTour: $currentTour) {
            id
            headshot {
              country
              countryFlag
              firstName
              lastName
            }
            standings {
              id
              title
              description
              total
              totalLabel
              rank
              owgr
              detailCopy
            }
            performance {
              tour
              season
              displaySeason
              stats {
                title
                value
                career
                wide
              }
            }
            snapshot {
              title
              value
              description
            }
            fedexFallStandings {
              id
              title
              description
              total
              totalLabel
              rank
              detailCopy
            }
          }
        }
        """

# Parse the API response
    async def parse_response(self, response_data: Dict) -> Optional[Dict]:
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning("No data found in response")
                return None

            overview_data = response_data.get('data', {}).get('playerProfileOverview')
            if not overview_data:
                self.logger.warning("No player profile overview data found")
                return None

            # Extract headshot data
            headshot = overview_data.get('headshot', {})
            
            # Process standings data
            standings = None
            if overview_data.get('standings'):
                try:
                    standings = StandingsData(
                        id=overview_data['standings'].get('id', ''),
                        title=overview_data['standings'].get('title', ''),
                        description=overview_data['standings'].get('description'),
                        total=overview_data['standings'].get('total'),
                        total_label=overview_data['standings'].get('totalLabel'),
                        rank=overview_data['standings'].get('rank'),
                        owgr=overview_data['standings'].get('owgr'),
                        detail_copy=overview_data['standings'].get('detailCopy')
                    )
                except Exception as e:
                    self.logger.error(f"Error processing standings data: {str(e)}")
            
            # Process FedEx Fall standings data
            fedex_fall_standings = None
            if overview_data.get('fedexFallStandings'):
                try:
                    fedex_fall_standings = StandingsData(
                        id=overview_data['fedexFallStandings'].get('id', ''),
                        title=overview_data['fedexFallStandings'].get('title', ''),
                        description=overview_data['fedexFallStandings'].get('description'),
                        total=overview_data['fedexFallStandings'].get('total'),
                        total_label=overview_data['fedexFallStandings'].get('totalLabel'),
                        rank=overview_data['fedexFallStandings'].get('rank'),
                        owgr=None,  # do not include OWGR for FedEx Fall standings not prseent in API
                        detail_copy=overview_data['fedexFallStandings'].get('detailCopy')
                    )
                except Exception as e:
                    self.logger.error(f"Error processing FedEx Fall standings data: {str(e)}")
        
            performance_data = []
            for performance in overview_data.get('performance', []):
                try:
                    stats = []
                    for stat in performance.get('stats', []):
                        stat_data = StatData(
                            title=stat.get('title', ''),
                            value=stat.get('value'),
                            career=stat.get('career'),
                            wide=stat.get('wide', False)
                        )
                        stats.append(stat_data)
                    
                    perf = PerformanceData(
                        tour=performance.get('tour', ''),
                        season=performance.get('season', ''),
                        display_season=performance.get('displaySeason', ''),
                        stats=stats
                    )
                    performance_data.append(perf)
                except Exception as e:
                    self.logger.error(f"Error processing performance data: {str(e)}")
                    continue
            
            snapshot_data = []
            for snapshot in overview_data.get('snapshot', []):
                try:
                    snap = SnapshotData(
                        title=snapshot.get('title', ''),
                        value=snapshot.get('value', ''),
                        description=snapshot.get('description')
                    )
                    snapshot_data.append(snap)
                except Exception as e:
                    self.logger.error(f"Error processing snapshot data: {str(e)}")
                    continue

            profile_overview = PlayerProfileOverviewData(
                player_id=overview_data.get('id', ''),
                first_name=headshot.get('firstName', ''),
                last_name=headshot.get('lastName', ''),
                country=headshot.get('country', ''),
                standings=standings,
                fedex_fall_standings=fedex_fall_standings,
                performance=performance_data,
                snapshot=snapshot_data
            )

            return profile_overview.dict()

        except Exception as e:
            self.logger.error(f"Error parsing player profile overview data: {str(e)}")
            return None

# Scraper - player  profile overview data
    async def scrape_player_profile_overview(self, player_id: str, tour_code: str = "R") -> Optional[Dict]:
        try:
            self.logger.info(f"Fetching profile overview for player {player_id} on tour {tour_code}")
            
            variables = {
                "playerId": player_id,
                "currentTour": tour_code
            }

            response_data = await self.make_request(self.query, variables)
            if not response_data:
                self.logger.warning(f"No response data for player {player_id}")
                return None

            return await self.parse_response(response_data)

        except Exception as e:
            self.logger.error(f"Error scraping player profile overview: {str(e)}")
            return None

async def scrape_multiple_players(self, player_ids: List[str], tour_code: str = "R") -> List[Dict]:
        try:
            self.logger.info(f"Batch processing profile overviews for {len(player_ids)} players")
            
            tasks = []
            for player_id in player_ids:
                tasks.append(self.scrape_player_profile_overview(player_id, tour_code))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and None values
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed: {str(result)}")
                elif result is not None:
                    valid_results.append(result)

            self.logger.info(f"Successfully scraped profile overviews for {len(valid_results)} players")
            return valid_results

        except Exception as e:
            self.logger.error(f"Error in batch scraping: {str(e)}")
            return []