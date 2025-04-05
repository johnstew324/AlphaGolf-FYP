from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field

# Pydantic model for player statistics
class PlayerStat(BaseModel):
    player_id: str
    name: str
    season: int
    stat_id: str
    title: str
    category: str
    rank: Optional[int] = None
    value: str
    supporting_stat_description: Optional[str] = ""
    supporting_stat_value: Optional[str] = ""
    supporting_value_description: Optional[str] = ""
    supporting_value: Optional[str] = ""
    collected_at: datetime = Field(default_factory=datetime.utcnow)

# Data scraper class for player statistics
class PlayerStatsScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.query = """
        query ProfileStatsFullV2($playerId: ID!, $year: Int) {
            playerProfileStatsFullV2(playerId: $playerId, year: $year) {
                playerProfileStatsFull {
                    tour
                    season
                    displaySeason
                    stats {
                        statId
                        rank
                        value
                        title
                        category
                        supportingStat {
                            description
                            value
                        }
                        supportingValue {
                            description
                            value
                        }
                    }
                }
            }
        }
        """

    async def parse_response(self, response_data: Dict, player_id: str, player_name: str, year: int) -> List[Dict]:
        try:
            stats_data_list = response_data.get('data', {}).get('playerProfileStatsFullV2', {}).get('playerProfileStatsFull', [])
            if not stats_data_list:
                self.logger.warning(f"No stats found for player {player_name} in year {year}")
                return []

            stats_data = stats_data_list[0]
            if stats_data.get('tour') != 'R':  # Ensure PGA Tour only
                self.logger.info(f"Skipping non-PGA Tour data for player {player_name}")
                return []

            formatted_stats = []
            for stat in stats_data.get('stats', []):
                try:
                    stat_dict = self._format_stat(stat, player_id, player_name, year)
                    # Validate using Pydantic model
                    validated_stat = PlayerStat(**stat_dict)
                    formatted_stats.append(validated_stat.dict())
                except Exception as e:
                    self.logger.error(f"Error formatting stat for {player_name}: {str(e)}")
                    continue

            return formatted_stats

        except Exception as e:
            self.logger.error(f"Error parsing response for {player_name}: {str(e)}")
            return []

    def _format_stat(self, stat: Dict, player_id: str, player_name: str, year: int) -> Dict:
        """Format individual stat data"""
        rank_value = stat.get('rank')
        # Convert '-' to None for rank
        if rank_value == '-':
            rank_value = None
        elif rank_value is not None:
            try:
                rank_value = int(rank_value)
            except (ValueError, TypeError):
                rank_value = None

        return {
            'player_id': player_id,
            'name': player_name,
            'season': year,
            'stat_id': stat.get('statId', ''),
            'title': stat.get('title', ''),
            'category': ', '.join(stat.get('category', [])),
            'rank': rank_value,
            'value': str(stat.get('value', '')),
            'supporting_stat_description': (stat.get('supportingStat') or {}).get('description', ''),
            'supporting_stat_value': str((stat.get('supportingStat') or {}).get('value', '')),
            'supporting_value_description': (stat.get('supportingValue') or {}).get('description', ''),
            'supporting_value': str((stat.get('supportingValue') or {}).get('value', ''))
        }

# Scrape player stats - for given year
    async def scrape_player_stats(self, player_id: str, player_name: str, year: int) -> Optional[Dict]:
        try:
            variables = {
                "playerId": player_id,
                "year": year
            }

            self.logger.info(f"Fetching stats for {player_name} ({year})")
            response_data = await self.make_request(self.query, variables)
            
            if not response_data:
                self.logger.warning(f"No response data for {player_name} in {year}")
                return []

            # Parse all stats
            stats_list = await self.parse_response(response_data, player_id, player_name, year)
            
            # Group stats into a single player object
            if stats_list:
                player_stats = {
                    'player_id': player_id,
                    'name': player_name,
                    'season': year,
                    'collected_at': datetime.utcnow(),
                    'stats': {}
                }
                
                # Group stats by category
                stat_categories = {}
                for stat in stats_list:
                    category = stat.get('category', 'Uncategorized')
                    if category not in stat_categories:
                        stat_categories[category] = []
                    
                    # Add the stat to the appropriate category
                    stat_categories[category].append({
                        'stat_id': stat.get('stat_id', ''),
                        'title': stat.get('title', ''),
                        'rank': stat.get('rank'),
                        'value': stat.get('value', ''),
                        'supporting_stat_description': stat.get('supporting_stat_description', ''),
                        'supporting_stat_value': stat.get('supporting_stat_value', ''),
                        'supporting_value_description': stat.get('supporting_value_description', ''),
                        'supporting_value': stat.get('supporting_value', '')
                    })
                
                player_stats['stats'] = stat_categories
                return player_stats
            
            return None

        except Exception as e:
            self.logger.error(f"Error scraping stats for {player_name}: {str(e)}")
            return None