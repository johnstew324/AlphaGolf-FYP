from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from data_scraper.base_scraper import BaseDataScraper
from pydantic import BaseModel, Field, validator
import asyncio

# Pydantic models for player career statistics, data, and dataset, row data, and table data
# found through api query to playerProfileCareer
from pydantic import BaseModel, validator, Field
from datetime import datetime

class AchievementData(BaseModel):
    title = None
    value = None

class YearlyStatData(BaseModel):
    year = None
    display_season = None
    tour_code = None
    events = None
    wins = None
    top10 = None
    top25 = None
    cuts_made = None
    second = None
    third = None
    official_money = None
    standings_points = None
    standings_rank = None
    withdrawn = None

    @validator('events', 'wins', 'top10', 'top25', 'cuts_made', 'second', 'third', 'withdrawn', 'standings_rank', pre=True)
    def validate_int_fields(cls, v):
        if v is None or v == '':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @validator('standings_points', pre=True)
    def validate_float_fields(cls, v):
        if v is None or v == '':
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

class TableRowData(BaseModel):
    row_title = None
    row_title_detail = None
    row_content = None
    second_content = None

    @validator('row_content', 'second_content', pre=True)
    def validate_content_fields(cls, v):
        if isinstance(v, list):
            return ' '.join(v)
        return v

class TableData(BaseModel):
    table_name = None
    table_detail = None
    rows = []

class PlayerCareerData(BaseModel):
    player_id = None
    tour_code = None
    events = None
    wins = None
    wins_title = None
    international_wins = None
    major_wins = None
    cuts_made = None
    runner_up = None
    second = None
    third = None
    top10 = None
    top25 = None
    official_money = None
    years = []
    achievements = []
    tables = []
    collected_at = Field(default_factory=datetime.utcnow)

    @validator('events', 'wins', 'international_wins', 'major_wins', 'cuts_made', 
               'runner_up', 'second', 'third', 'top10', 'top25', pre=True)
    def validate_int_fields(cls, v):
        if v is None or v == '':
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

class PlayerCareerScraper(BaseDataScraper):
    def __init__(self, url, headers):
        super().__init__(url, headers)
        self.query = """
        query PlayerProfileCareer($playerId: String!, $tourCode: TourCode) {
          playerProfileCareer(playerId: $playerId, tourCode: $tourCode) {
            playerId
            tourCode
            events
            wins
            winsTitle
            internationalWins
            majorWins
            cutsMade
            runnerUp
            second
            third
            top10
            top25
            years {
              cutsMade
              displaySeason
              events
              officialMoney
              second
              standingsPoints
              standingsRank
              third
              top10
              top25
              tourCode
              wins
              withdrawn
              year
            }
            officialMoney
            tourPills {
              tourCode
              displayName
            }
            achievements {
              title
              value
            }
            tables {
              tableName
              tableDetail
              rows {
                rowTitle
                rowTitleDetail
                rowContent
                secondContent
              }
            }
          }
        }
        """

# parse api reponse
    async def parse_response(self, response_data):
        try:
            if not response_data or 'data' not in response_data:
                self.logger.warning("No data found in response")
                return None

            career_data = response_data.get('data', {}).get('playerProfileCareer')
            if not career_data:
                self.logger.warning("No player career data found")
                return None
            yearly_stats = []
            for year in career_data.get('years', []):
                try:
                    year_data = YearlyStatData(
                        year=year.get('year'),
                        display_season=year.get('displaySeason', ''),
                        tour_code=year.get('tourCode', ''),
                        events=year.get('events'),
                        wins=year.get('wins'),
                        top10=year.get('top10'),
                        top25=year.get('top25'),
                        cuts_made=year.get('cutsMade'),
                        second=year.get('second'),
                        third=year.get('third'),
                        official_money=year.get('officialMoney'),
                        standings_points=year.get('standingsPoints'),
                        standings_rank=year.get('standingsRank'),
                        withdrawn=year.get('withdrawn')
                    )
                    yearly_stats.append(year_data)
                except Exception as e:
                    self.logger.error(f"Error processing year data: {str(e)}")
                    continue
                
                
            
            achievements = []
            for achievement in career_data.get('achievements', []):
                try:
                    achievement_data = AchievementData(
                        title=achievement.get('title', ''),
                        value=achievement.get('value', '')
                    )
                    achievements.append(achievement_data)
                except Exception as e:
                    self.logger.error(f"Error processing achievement data: {str(e)}")
                    continue


            tables = []
            for table in career_data.get('tables', []):
                try:
                    rows = []
                    for row in table.get('rows', []):
                        # Debug the row data format
                        self.logger.debug(f"Processing row: {row}")
                        
                        # Convert list values to strings if needed
                        row_content = row.get('rowContent', '')
                        second_content = row.get('secondContent')
                        
                        row_data = TableRowData(
                            row_title=row.get('rowTitle', ''),
                            row_title_detail=row.get('rowTitleDetail'),
                            row_content=row_content,
                            second_content=second_content
                        )
                        rows.append(row_data)
                        
                    table_data = TableData(
                        table_name=table.get('tableName', ''),
                        table_detail=table.get('tableDetail'),
                        rows=rows
                    )
                    tables.append(table_data)
                except Exception as e:
                    self.logger.error(f"Error processing table data: {str(e)}")
                    continue

           
           
            player_career = PlayerCareerData(
                player_id=career_data.get('playerId', ''),
                tour_code=career_data.get('tourCode', ''),
                events=career_data.get('events'),
                wins=career_data.get('wins'),
                wins_title=career_data.get('winsTitle'),
                international_wins=career_data.get('internationalWins'),
                major_wins=career_data.get('majorWins'),
                cuts_made=career_data.get('cutsMade'),
                runner_up=career_data.get('runnerUp'),
                second=career_data.get('second'),
                third=career_data.get('third'),
                top10=career_data.get('top10'),
                top25=career_data.get('top25'),
                official_money=career_data.get('officialMoney'),
                years=yearly_stats,
                achievements=achievements,
                tables=tables
            )

            return player_career.dict()

        except Exception as e:
            self.logger.error(f"Error parsing player career data: {str(e)}")
            return None

#scraper class - player career statistics
    async def scrape_player_career(self, player_id, tour_code = "R") :
        
        try:
            self.logger.info(f"Fetching career stats for player {player_id} on tour {tour_code}")
            
            variables = {
                "playerId": player_id,
                "tourCode": tour_code
            }

            response_data = await self.make_request(self.query, variables)
            if not response_data:
                self.logger.warning(f"No response data for player {player_id}")
                return None

            return await self.parse_response(response_data)

        except Exception as e:
            self.logger.error(f"Error scraping player career stats: {str(e)}")
            return None

# batch processing of multiple players
    async def scrape_multiple_players(self, player_ids, tour_code = "R"):
        try:
            self.logger.info(f"Batch processing career stats for {len(player_ids)} players")
            tasks = []
            for player_id in player_ids:
                tasks.append(self.scrape_player_career(player_id, tour_code))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed: {str(result)}")
                elif result is not None:
                    valid_results.append(result)

            self.logger.info(f"Successfully scraped career data for {len(valid_results)} players")
            return valid_results

        except Exception as e:
            self.logger.error(f"Error in batch scraping: {str(e)}")
            return []