from datetime import datetime
from typing import Dict, List, Optional
import asyncio
from data_scraper.base_scraper import BaseDataScraper

# Data scraper class for shot data
class ShotDataScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query CourseData($tournamentId: String!, $course: Int!, $hole: Int!) {
            scatterData(tournamentId: $tournamentId, course: $course, hole: $hole) {
                id
                tournamentId
                hole
                rounds {
                    display
                    num
                    strokes {
                        strokeNumber
                        playerShots {
                            result
                            player {
                                id
                                name
                            }
                            shotCoords {
                                green {
                                    landscapeCoords {
                                        x
                                        y
                                    }
                                    portraitCoords {
                                        x
                                        y
                                    }
                                }
                                overview {
                                    landscapeCoords {
                                        x
                                        y
                                    }
                                    portraitCoords {
                                        x
                                        y
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

# Parse the API response
    async def parse_response(self, response_data: Dict) -> Optional[Dict]:
        if not response_data or 'data' not in response_data:
            return None
        return response_data.get('data', {}).get('scatterData')

# Scraper -  shot data for all holes 
    async def scrape_shot_data(self, tournament_id: str, course_id: int) -> Optional[Dict]:
        try:
            tasks = []
            for hole in range(1, 19):
                self.logger.info(f"Processing hole {hole}...")
                variables = {
                    "tournamentId": tournament_id,
                    "course": course_id,
                    "hole": hole
                }
                tasks.append(self.process_hole(variables))
            
            all_holes_data = await asyncio.gather(*tasks)
            all_holes_data = [data for data in all_holes_data if data is not None]
            
            if not all_holes_data:
                return None
                
            shot_data = {
                'tournament_id': tournament_id,
                'course_id': course_id,
                'holes_data': all_holes_data,
                'player_stats': await self.calculate_player_stats(all_holes_data),
                'collected_at': datetime.utcnow()
            }
            
            return shot_data
            
        except Exception as e:
            self.logger.error(f"Error scraping shot data: {str(e)}")
            return None

    async def process_hole(self, variables: Dict) -> Optional[Dict]:
        try:
            response_data = await self.make_request(self.query, variables)
            if response_data and 'data' in response_data:
                return self.parse_hole_shots(response_data['data']['scatterData'])
            return None
        except Exception as e:
            self.logger.error(f"Error processing hole {variables.get('hole')}: {str(e)}")
            return None

    def parse_hole_shots(self, hole_data: Dict) -> Optional[Dict]:
        if not hole_data:
            return None
            
        hole_shots = {
            'hole_number': hole_data.get('hole'),
            'hole_id': hole_data.get('id'),
            'rounds': []
        }
        
        for round_data in hole_data.get('rounds', []):
            round_info = {
                'round_number': round_data.get('num'),
                'round_display': round_data.get('display'),
                'strokes': []
            }
            
            for stroke in round_data.get('strokes', []):
                stroke_info = {
                    'stroke_number': stroke.get('strokeNumber'),
                    'player_shots': []
                }
                
                for shot in stroke.get('playerShots', []):
                    player_shot = {
                        'player': {
                            'id': shot['player']['id'],
                            'name': shot['player']['name']
                        },
                        'result': shot.get('result'),
                        'coordinates': {
                            'green': {
                                'landscape': shot['shotCoords']['green']['landscapeCoords'],
                                'portrait': shot['shotCoords']['green']['portraitCoords']
                            },
                            'overview': {
                                'landscape': shot['shotCoords']['overview']['landscapeCoords'],
                                'portrait': shot['shotCoords']['overview']['portraitCoords']
                            }
                        }
                    }
                    stroke_info['player_shots'].append(player_shot)
                
                round_info['strokes'].append(stroke_info)
            
            hole_shots['rounds'].append(round_info)
        
        return hole_shots


# calculate player stats for all holes
    async def calculate_player_stats(self, holes_data: List[Dict]) -> Dict:
        player_stats = {}
        
        for hole in holes_data:
            for round_data in hole.get('rounds', []):
                for stroke in round_data.get('strokes', []):
                    for player_shot in stroke.get('player_shots', []):
                        player_id = player_shot['player']['id']
                        
                        if player_id not in player_stats:
                            player_stats[player_id] = {
                                'player_name': player_shot['player']['name'],
                                'holes_played': set(),
                                'rounds': {},
                                'results': {
                                    'BIRDIE': 0,
                                    'PAR': 0,
                                    'BOGEY': 0,
                                    'EAGLE': 0,
                                    'DOUBLE_BOGEY': 0
                                }
                            }
                        
                        player_stats[player_id]['holes_played'].add(hole['hole_number'])
                        
                        round_num = str(round_data['round_number'])
                        if round_num not in player_stats[player_id]['rounds']:
                            player_stats[player_id]['rounds'][round_num] = {
                                'strokes': 0,
                                'results': {
                                    'BIRDIE': 0,
                                    'PAR': 0,
                                    'BOGEY': 0,
                                    'EAGLE': 0,
                                    'DOUBLE_BOGEY': 0
                                }
                            }
                        
                        result = player_shot.get('result')
                        if result:
                            player_stats[player_id]['results'][result] += 1
                            player_stats[player_id]['rounds'][round_num]['results'][result] += 1
        
        formatted_stats = {}
        for player_id in player_stats:
            player_stats[player_id]['holes_played'] = len(player_stats[player_id]['holes_played'])
            formatted_stats[str(player_id)] = player_stats[player_id]
        
        return formatted_stats