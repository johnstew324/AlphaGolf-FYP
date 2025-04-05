from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field, validator


# Pydantic models for tournament history statistics, data, and dataset
class TournamentHistoryResult(BaseModel):
    tournament_id: str
    end_date: str
    name: str
    tour_code: str
    position: str
    score: Optional[Union[float, str]] = None
    season: int

    @validator('score', pre=True)
    def validate_score(cls, v):
        if v is None or v == '':
            return None
        if v == 'E':  # Even par
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return str(v)

class StrokesGainedHistory(BaseModel):
    stat_id: str
    stat_value: Optional[float]
    stat_color: Optional[str]

    @validator('stat_value', pre=True)
    def validate_stat_value(cls, v):
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

class PlayerTournamentHistory(BaseModel):
    player_id: str
    total_rounds: int
    tournament_results: List[TournamentHistoryResult]
    strokes_gained: List[StrokesGainedHistory]
    strokes_gained_header: List[str]

class TournamentHistoryStatsData(BaseModel):
    tournament_id: str
    field_stat_type: str
    players: List[PlayerTournamentHistory]
    stat_headers: List[str] = Field(default_factory=list)  # Default to empty list
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('stat_headers', pre=True)
    def validate_stat_headers(cls, v):
        if v is None:
            return []  # Return empty list if None
        return v
# Data scraper class for tournament history statistics
class TournamentHistoryStatsScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query FieldStats($tournamentId: ID!, $fieldStatType: FieldStatType!) {
          fieldStats(tournamentId: $tournamentId, fieldStatType: $fieldStatType) {
            tournamentId
            fieldStatType
            players {
              __typename
              ... on FieldStatCurrentForm {
                playerId
                totalRounds
                tournamentResults {
                  tournamentId
                  endDate
                  name
                  tourCode
                  position
                  score
                  season
                }
                strokesGained {
                  statId
                  statValue
                  statColor
                }
                strokesGainedHeader
              }
            }
            statHeaders
          }
        }
        """

    async def parse_response(self, response_data: Dict, tournament_id: str) -> Optional[Dict]:
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id}")
                return None

            field_stats = response_data.get('data', {}).get('fieldStats')
            if not field_stats:
                self.logger.warning(f"No field stats found for tournament {tournament_id}")
                return None

            # Process and validate player data
            processed_players = []
            for player in field_stats.get('players', []):
                try:
                    # Process tournament results
                    tournament_results = [
                        TournamentHistoryResult(
                            tournament_id=result['tournamentId'],
                            end_date=result['endDate'],
                            name=result['name'],
                            tour_code=result['tourCode'],
                            position=result['position'],
                            score=result.get('score'),
                            season=result['season']
                        ) for result in player.get('tournamentResults', [])
                    ]

                    # Process strokes gained
                    strokes_gained = [
                        StrokesGainedHistory(
                            stat_id=sg['statId'],
                            stat_value=sg.get('statValue'),
                            stat_color=sg.get('statColor')
                        ) for sg in player.get('strokesGained', [])
                    ]

                    player_data = PlayerTournamentHistory(
                        player_id=player['playerId'],
                        total_rounds=player['totalRounds'],
                        tournament_results=tournament_results,
                        strokes_gained=strokes_gained,
                        strokes_gained_header=player.get('strokesGainedHeader', [])
                    )
                    processed_players.append(player_data)

                except Exception as e:
                    self.logger.error(f"Error processing player data: {str(e)}")
                    continue

            # Ensure stat_headers is at least an empty list if None
            stat_headers = field_stats.get('statHeaders')
            if stat_headers is None:
                stat_headers = []

            # Validate complete dataset
            history_data = TournamentHistoryStatsData(
                tournament_id=tournament_id,
                field_stat_type="TOURNAMENT_HISTORY",
                players=processed_players,
                stat_headers=stat_headers
            )

            return history_data.dict()

        except Exception as e:
            self.logger.error(f"Error parsing tournament history data: {str(e)}")
            return None


    async def scrape_tournament_history_stats(self, tournament_id: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Fetching tournament history stats for tournament {tournament_id}")
            variables = {
                "tournamentId": tournament_id,
                "fieldStatType": "TOURNAMENT_HISTORY"
            }

            response_data = await self.make_request(self.query, variables)
            if not response_data:
                self.logger.warning(f"No response data for tournament {tournament_id}")
                return None
            return await self.parse_response(response_data, tournament_id)

        except Exception as e:
            self.logger.error(f"Error scraping tournament history stats: {str(e)}")
            return None