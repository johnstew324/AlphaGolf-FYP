from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field, validator


 # Pydantic models for current form statistics, data, and dataset
class TournamentResult(BaseModel):
    tournament_id: str
    end_date: str
    name: str
    tour_code: str
    position: str
    score: Optional[Union[float, str]] = None  # Allow both float and string
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
            return str(v)  # Keep as string if can't convert to float

class StrokesGained(BaseModel):
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

class PlayerCurrentForm(BaseModel):
    player_id: str
    total_rounds: int
    tournament_results: List[TournamentResult]
    strokes_gained: List[StrokesGained]
    strokes_gained_header: List[str]


class CurrentFormData(BaseModel):
    tournament_id: str
    field_stat_type: str
    players: List[PlayerCurrentForm]
    stat_headers: List[str] = Field(default_factory=list)  # Default to empty list
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('stat_headers', pre=True)
    def validate_stat_headers(cls, v):
        if v is None:
            return []  # Return empty list if None
        return v
    
#Data scraper class for current form statistics
# query from pga tour API
class CurrentFormScraper(BaseDataScraper):
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

#Parse API response
    async def parse_response(self, response_data, tournament_id):
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id}")
                return None

            field_stats = response_data.get('data', {}).get('fieldStats')
            if not field_stats:
                self.logger.warning(f"No field stats found for tournament {tournament_id}")
                return None
            
            processed_players = []
            for player in field_stats.get('players', []):
                try:
                    tournament_results = [
                        TournamentResult(
                            tournament_id=result['tournamentId'],
                            end_date=result['endDate'],
                            name=result['name'],
                            tour_code=result['tourCode'],
                            position=result['position'],
                            score=result.get('score'),
                            season=result['season']
                        ) for result in player.get('tournamentResults', [])
                    ]

                    strokes_gained = [
                        StrokesGained(
                            stat_id=sg['statId'],
                            stat_value=sg.get('statValue'),
                            stat_color=sg.get('statColor')
                        ) for sg in player.get('strokesGained', [])
                    ]

                    player_data = PlayerCurrentForm(
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

            stat_headers = field_stats.get('statHeaders')
            if stat_headers is None:
                stat_headers = []

            current_form_data = CurrentFormData(
                tournament_id=tournament_id,
                field_stat_type="CURRENT_FORM",
                players=processed_players,
                stat_headers=stat_headers
            )

            return current_form_data.dict()

        except Exception as e:
            self.logger.error(f"Error parsing current form data: {str(e)}")
            return None

# Scraper  current form statistics
    async def scrape_current_form(self, tournament_id: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Fetching current form stats for tournament {tournament_id}")
            
            variables = {
                "tournamentId": tournament_id,
                "fieldStatType": "CURRENT_FORM"
            }

            response_data = await self.make_request(self.query, variables)
            if not response_data:
                self.logger.warning(f"No response data for tournament {tournament_id}")
                return None

            return await self.parse_response(response_data, tournament_id)

        except Exception as e:
            self.logger.error(f"Error scraping current form stats: {str(e)}")
            return None