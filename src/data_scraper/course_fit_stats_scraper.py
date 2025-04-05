from typing import Dict, List, Optional, Any
from datetime import datetime
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field, validator
import asyncio


# Pydantic models for course fit statistics, data, and dataset.
class CourseFitStat(BaseModel):
    header: str
    value: Optional[float] = None
    rank: Optional[int] = None
    color: Optional[str] = None
    @validator('value', 'rank', pre=True)
    def handle_dash(cls, v):
        return None if v == '-' else v

class PlayerCourseFit(BaseModel):
    player_id: str
    total_rounds: int
    score: float
    stats: List[CourseFitStat]

class CourseFitData(BaseModel):
    tournament_id: str
    field_stat_type: str
    stat_headers: List[str]
    players: List[PlayerCourseFit]
    collected_at: datetime = Field(default_factory=datetime.utcnow)

# Data scraper class for course fit statistics 
class CourseFitStatsScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query FieldStats($tournamentId: ID!, $fieldStatType: FieldStatType!) {
            fieldStats(tournamentId: $tournamentId, fieldStatType: $fieldStatType) {
                tournamentId
                fieldStatType
                players {
                    ... on FieldStatCourseFit {
                        playerId
                        totalRounds
                        score
                        stats {
                            statValue
                            statRank
                            statColor
                        }
                    }
                }
                statHeaders
            }
        }
        """

    # Parse and validate the API response
    async def parse_response(self, response_data: Dict, tournament_id: str, field_stat_type: str) -> Optional[Dict]:
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id}")
                return None

            field_stats = response_data.get('data', {}).get('fieldStats')
            if not field_stats:
                self.logger.warning(f"No field stats found for tournament {tournament_id}")
                return None

            #process and validate player data
            processed_players = []
            for player in field_stats.get('players', []):
                try:
                    player_stats = []
                    for stat, header in zip(player.get('stats', []), field_stats.get('statHeaders', [])):
                        stat_data = CourseFitStat(
                            header=header,
                            value=stat.get('statValue'),
                            rank=stat.get('statRank'),
                            color=stat.get('statColor')
                        )
                        player_stats.append(stat_data)

                    player_data = PlayerCourseFit(
                        player_id=player['playerId'],
                        total_rounds=player['totalRounds'],
                        score=player['score'],
                        stats=player_stats
                    )
                    processed_players.append(player_data)

                except Exception as e:
                    self.logger.error(f"Error processing player data: {str(e)}")
                    continue

            course_fit_data = CourseFitData(
                tournament_id=tournament_id,
                field_stat_type=field_stat_type,
                stat_headers=field_stats.get('statHeaders', []),
                players=processed_players
            )

            return course_fit_data.dict()

        except Exception as e:
            self.logger.error(f"Error parsing course fit data: {str(e)}")
            return None

        # Scrape course fit statistics for a tournament 
        # retuns parsed and validated course fit statistics 
    async def scrape_coursefit_stats(self, tournament_id: str, field_stat_type: str = "COURSE_FIT") -> Optional[Dict]:
        try:
            self.logger.info(f"Fetching course fit stats for tournament {tournament_id}")
            
            variables = {"tournamentId": tournament_id,"fieldStatType": field_stat_type}

            response_data = await self.make_request(self.query, variables)
            if not response_data:
                self.logger.warning(f"No response data for tournament {tournament_id}")
                return None
            return await self.parse_response(response_data, tournament_id, field_stat_type)

        except Exception as e:
            self.logger.error(f"Error scraping course fit stats: {str(e)}")
            return None

    # Scrape course fit statistics for multiple tournaments concurrently
    # returns a list of processed course fit statistics
    async def scrape_multiple_tournaments(self, tournament_ids: List[str], field_stat_type: str = "COURSE_FIT") -> List[Dict]:
        try:
            self.logger.info(f"Batch processing {len(tournament_ids)} tournaments")
            
            tasks = [
                self.scrape_coursefit_stats(tournament_id, field_stat_type)
                for tournament_id in tournament_ids
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed: {str(result)}")
                    continue
                if result is not None:
                    valid_results.append(result)
            return valid_results

        except Exception as e:
            self.logger.error(f"Error in batch scraping: {str(e)}")
            return []