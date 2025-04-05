from data_scraper.base_scraper import BaseDataScraper
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio

# Pydantic models for scorecard data 
class HoleScore(BaseModel):
    holeNumber: int
    par: int
    score: int
    status: str
    yardage: int
    roundScore: Optional[int] = None
    
    # Custom alias generator to remove underscores
    class Config:
        alias_generator = lambda x: x.replace('_', '')

class NineHoles(BaseModel):
    holes: List[HoleScore]
    totalLabel: str
    parTotal: int
    total: int

class RoundScore(BaseModel):
    roundNumber: int
    complete: bool
    currentHole: Optional[int]
    currentRound: bool
    firstNine: NineHoles
    secondNine: NineHoles
    courseName: str
    parTotal: int
    total: int
    scoreToPar: int

class PlayerScorecard(BaseModel):
    tournament_name: str
    id: str
    currentRound: int
    backNine: bool
    groupNumber: Optional[int]
    player_id: str
    player_name: str
    player_country: Optional[str]
    roundScores: List[RoundScore]       
    currentHole: Optional[int]
    playerState: str
    collected_at: datetime = Field(default_factory=datetime.utcnow)

class ScorecardScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query ScorecardData($tournamentId: ID!, $playerId: ID!) {
          scorecardV3(tournamentId: $tournamentId, playerId: $playerId) {
            tournamentName
            id
            currentRound
            backNine
            groupNumber
            player {
              id
              displayName
              country
            }
            roundScores {
              roundNumber
              complete
              currentHole
              currentRound
              firstNine {
                holes {
                  holeNumber
                  par
                  score
                  status
                  yardage
                  roundScore
                }
                totalLabel
                parTotal
                total
              }
              secondNine {
                holes {
                  holeNumber
                  par
                  score
                  status
                  yardage
                  roundScore
                }
                totalLabel
                parTotal
                total
              }
              courseName
              parTotal
              total
              scoreToPar
            }
            currentHole
            playerState
          }
        }
        """

    async def parse_response(self, response_data: Dict) -> Optional[Dict]:
        if not response_data or 'data' not in response_data:
            return None
        return response_data.get('data', {}).get('scorecardV3')

#  scraper - scorecard data 
    async def scrape_scorecard(self, tournament_id: str, player_id: str) -> Optional[Dict]:
        """Scrape scorecard data for a single player"""
        try:
            variables = {
                "tournamentId": tournament_id,
                "playerId": player_id
            }
            
            response_data = await self.make_request(self.query, variables)
            if not response_data:
                return None

            scorecard_data = await self.parse_response(response_data)
            if not scorecard_data:
                return None

            return {
                "tournament_name": scorecard_data['tournamentName'],
                "id": scorecard_data['id'],
                "currentRound": scorecard_data['currentRound'],
                "backNine": scorecard_data['backNine'],
                "groupNumber": scorecard_data.get('groupNumber'),
                "player_id": scorecard_data['player']['id'],
                "player_name": scorecard_data['player']['displayName'],
                "player_country": scorecard_data['player'].get('country'),
                "roundScores": scorecard_data['roundScores'],
                "currentHole": scorecard_data.get('currentHole'),
                "playerState": scorecard_data['playerState'],
                "collected_at": datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Error scraping scorecard for player {player_id}: {str(e)}")
            return None

        # multiple scorecards for a tournament
    async def scrape_multiple_scorecards(self, tournament_id: str, player_ids: List[str]) -> List[Dict]:
        try:
            tasks = [
                self.scrape_scorecard(tournament_id, player_id)
                for player_id in player_ids
            ]
            results = await asyncio.gather(*tasks)
            valid_results = [result for result in results if result is not None]
            
            
            if not valid_results:
                self.logger.warning(f"No valid scorecards found for tournament {tournament_id}")
            return valid_results
            
        except Exception as e:
            self.logger.error(f"Error in batch scraping: {str(e)}")
            return []