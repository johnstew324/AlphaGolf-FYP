from data_scraper.base_scraper import BaseDataScraper
from datetime import datetime
import asyncio
from typing import Dict, List, Optional

# Pydantic models for course statistics
class CourseStatsScraper(BaseDataScraper):
    def __init__(self, url: str, headers: Dict[str, str]):
        super().__init__(url, headers)
        self.course_stats_query = """
        query CourseStats($tournamentId: ID!) {
          courseStats(tournamentId: $tournamentId) {
            tournamentId
            courses {
              tournamentId
              courseId
              courseName
              courseCode
              holeDetailsAvailability
              par
              yardage
              hostCourse
              roundHoleStats {
                roundHeader
                roundNum
                live
                holeStats {
                  __typename
                  ... on CourseHoleStats {
                    courseHoleNum
                    parValue
                    yards
                    scoringAverage
                    scoringAverageDiff
                    scoringDiffTendency
                    eagles
                    birdies
                    pars
                    bogeys
                    doubleBogey
                    rank
                    live
                    pinGreen {
                      leftToRightCoords {
                        x
                        y
                        z
                      }
                      bottomToTopCoords {
                        x
                        y
                        z
                      }
                    }
                  }
                }
              }
              courseOverview {
                id
                name
                city
                state
                country
                overview {
                  label
                  value
                  detail
                  secondaryDetail
                }
              }
            }
          }
        }
        """
        
# Query to get the data structure for "aboutThisHole"
        self.hole_details_query = """
        query HoleDetails($tournamentId: ID!, $courseId: ID!, $hole: Int!) {
            holeDetails(tournamentId: $tournamentId, courseId: $courseId, hole: $hole) {
                id
                tournamentId
                statsAvailability
                holeNum
                courseId
                statsSummary {
                    eagles
                    birdies
                    pars
                    bogeys
                    doubleBogeys
                }
                holeInfo {
                    par
                    yards
                    scoringAverageDiff
                    aboutThisHole
                    pinGreen {
                        leftToRightCoords {
                            x
                            y
                            z
                        }
                        bottomToTopCoords {
                            x
                            y
                            z
                        }
                    }
                }
            }
        }
        """

    def _safe_int(self, value, default=0):
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
            
    def _safe_float(self, value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


    async def parse_response(self, response_data: Dict) -> Optional[Dict]:
        if not response_data or 'data' not in response_data:
            return None
            
        return response_data.get('data', {}).get('courseStats', {})


    async def get_tournament_courses(self, tournament_id: str) -> List[Dict]:
        try:
            variables = {"tournamentId": tournament_id}
            response_data = await self.make_request(self.course_stats_query, variables)
            
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id}")
                return []
                
                
            courses = response_data['data']['courseStats'].get('courses', [])
            return [{
                'tournament_id': tournament_id,
                'course_id': course['courseId'],
                'course_name': course['courseName'],
                'course_code': course.get('courseCode', ''),
                'host_course': course.get('hostCourse', False),
                'par': self._safe_int(course.get('par'), 0),
                'yardage': self._safe_int(course.get('yardage'), 0)
            } for course in courses]
            
            
        except Exception as e:
            self.logger.error(f"Error getting tournament courses: {str(e)}")
            return []


    async def get_hole_details(self, tournament_id: str, course_id: str, hole_number: int) -> Optional[Dict]:
        try:
            variables = {"tournamentId": tournament_id,
                "courseId": course_id,
                "hole": hole_number
            }
            
            response_data = await self.make_request(self.hole_details_query, variables)
            
            if not response_data or 'data' not in response_data:
                return None
                
            hole_details = response_data.get('data', {}).get('holeDetails', {})
            if not hole_details:
                return None
                
            hole_info = hole_details.get('holeInfo', {})
            
            return {
                'about_this_hole': hole_info.get('aboutThisHole', ''),
                'par': self._safe_int(hole_info.get('par'), 0),
                'yards': self._safe_int(hole_info.get('yards'), 0),
                'scoring_average_diff': self._safe_float(hole_info.get('scoringAverageDiff'), 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting hole details: {str(e)}")
            return None

    async def scrape_course_stats(self, tournament_id: str, course_id: str) -> Optional[Dict]:
        try:
            variables = {"tournamentId": tournament_id}
            response_data = await self.make_request(self.course_stats_query, variables)
            
            if not response_data or 'data' not in response_data:
                self.logger.warning(f"No data found for tournament {tournament_id}")
                return None
                
            courses = response_data['data']['courseStats'].get('courses', [])
            if not courses:
                self.logger.warning(f"No courses found for tournament {tournament_id}")
                return None
                
            # Find the specific course
            course_data = None
            for course in courses:
                if course['courseId'] == course_id:
                    course_data = course
                    break
                    
            if not course_data:
                self.logger.warning(f"Course {course_id} not found for tournament {tournament_id}")
                return None
                
            processed_data = {
                'tournament_id': tournament_id,
                'course_id': course_id,
                'course_name': course_data.get('courseName', ''),
                'course_code': course_data.get('courseCode', ''),
                'par': self._safe_int(course_data.get('par'), 0),
                'yardage': self._safe_int(course_data.get('yardage'), 0),
                'host_course': course_data.get('hostCourse', False),
                'collected_at': datetime.utcnow(),
                'rounds': []
            }
            
            round_hole_stats = course_data.get('roundHoleStats', [])
            
            hole_details_cache = {}
            
            for round_stat in round_hole_stats:
                round_num = self._safe_int(round_stat.get('roundNum'))
                if not round_num:
                    continue
                    
                round_data = {
                    'round_number': round_num,
                    'round_header': round_stat.get('roundHeader', ''),
                    'live': round_stat.get('live', False),
                    'holes': []
                }
                
                # Process holes for this round
                hole_stats = round_stat.get('holeStats', [])
                for hole_stat in hole_stats:
                    if hole_stat.get('__typename') != 'CourseHoleStats':
                        continue
                        
                    hole_number = self._safe_int(hole_stat.get('courseHoleNum'), 0)
                
                    if hole_number not in hole_details_cache:
                        hole_details = await self.get_hole_details(tournament_id, course_id, hole_number)
                        hole_details_cache[hole_number] = hole_details or {}
                    
                    pin_green = hole_stat.get('pinGreen', {})
                    left_to_right = pin_green.get('leftToRightCoords', {})
                    bottom_to_top = pin_green.get('bottomToTopCoords', {})
                    
                    # Create the hole data object
                    hole_data = {
                        'hole_number': hole_number,
                        'par': self._safe_int(hole_stat.get('parValue'), 0),
                        'yards': self._safe_int(hole_stat.get('yards'), 0),
                        'scoring_average': self._safe_float(hole_stat.get('scoringAverage'), 0.0),
                        'scoring_average_diff': self._safe_float(hole_stat.get('scoringAverageDiff'), 0.0),
                        'scoring_diff_tendency': hole_stat.get('scoringDiffTendency', ''),
                        'eagles': self._safe_int(hole_stat.get('eagles'), 0),
                        'birdies': self._safe_int(hole_stat.get('birdies'), 0),
                        'pars': self._safe_int(hole_stat.get('pars'), 0),
                        'bogeys': self._safe_int(hole_stat.get('bogeys'), 0),
                        'double_bogeys': self._safe_int(hole_stat.get('doubleBogey'), 0),
                        'rank': self._safe_int(hole_stat.get('rank'), 0),
                        'live': hole_stat.get('live', False),
                        'pin_location': {
                            'left_to_right': {
                                'x': left_to_right.get('x'),
                                'y': left_to_right.get('y'),
                                'z': left_to_right.get('z')
                            },
                            'bottom_to_top': {
                                'x': bottom_to_top.get('x'),
                                'y': bottom_to_top.get('y'),
                                'z': bottom_to_top.get('z')
                            }
                        }
                    }
                    
                # Add hole details
                    hole_detail = hole_details_cache.get(hole_number, {})
                    if hole_detail:
                        hole_data['about_this_hole'] = hole_detail.get('about_this_hole', '')
                    
                    round_data['holes'].append(hole_data)
                
                processed_data['rounds'].append(round_data)
                
        # Add course overview 
            overview = course_data.get('courseOverview', {})
            if overview:
                processed_data['overview'] = {
                    'name': overview.get('name', ''),
                    'city': overview.get('city', ''),
                    'state': overview.get('state', ''),
                    'country': overview.get('country', ''),
                    'details': []
                }
                
                for item in overview.get('overview', []):
                    processed_data['overview']['details'].append({
                        'label': item.get('label', ''),
                        'value': item.get('value', ''),
                        'detail': item.get('detail', ''),
                        'secondary_detail': item.get('secondaryDetail', '')
                    })
            
            processed_data['course_summary'] = self.calculate_course_summary(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error scraping course stats: {str(e)}")
            return None

# Calculate summary statistics for the course
    def calculate_course_summary(self, course_data: Dict) -> Dict:
        total_stats = {
            'eagles': 0,
            'birdies': 0,
            'pars': 0,
            'bogeys': 0,
            'double_bogeys': 0,
            'total_yards': 0,
            'total_par': 0,
            'rounds_complete': 0
        }
        
        all_holes = {}
        
        rounds_summary = {}
        for round_data in course_data.get('rounds', []):
            round_num = self._safe_int(round_data.get('round_number'))
            if not round_num:
                continue
                
            round_summary = {
                'header': round_data.get('round_header', ''),
                'eagles': 0,
                'birdies': 0,
                'pars': 0,
                'bogeys': 0,
                'double_bogeys': 0,
                'hole_count': 0
            }
            
            for hole in round_data.get('holes', []):
                hole_number = hole.get('hole_number')
                if hole_number not in all_holes:
                    all_holes[hole_number] = hole
                
                # Add to round summary
                round_summary['eagles'] += self._safe_int(hole.get('eagles'), 0)
                round_summary['birdies'] += self._safe_int(hole.get('birdies'), 0)
                round_summary['pars'] += self._safe_int(hole.get('pars'), 0)
                round_summary['bogeys'] += self._safe_int(hole.get('bogeys'), 0)
                round_summary['double_bogeys'] += self._safe_int(hole.get('double_bogeys'), 0)
                round_summary['hole_count'] += 1
                
            rounds_summary[str(round_num)] = round_summary
            
            if round_summary['hole_count'] == 18:
                total_stats['rounds_complete'] += 1
        
        for hole_number, hole in all_holes.items():
            total_stats['eagles'] += self._safe_int(hole.get('eagles'), 0)
            total_stats['birdies'] += self._safe_int(hole.get('birdies'), 0)
            total_stats['pars'] += self._safe_int(hole.get('pars'), 0)
            total_stats['bogeys'] += self._safe_int(hole.get('bogeys'), 0)
            total_stats['double_bogeys'] += self._safe_int(hole.get('double_bogeys'), 0)
            total_stats['total_yards'] += self._safe_int(hole.get('yards'), 0)
            total_stats['total_par'] += self._safe_int(hole.get('par'), 0)
        
        return {
            **total_stats,
            'rounds_summary': rounds_summary
        }
        
        
# Test function to scrape course stats without storing in database
    async def test_scrape_course_stats(self, tournament_id: str, course_id: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Test scraping course stats for tournament {tournament_id}, course {course_id}")
            course_data = await self.scrape_course_stats(tournament_id, course_id)
            
            if course_data:
                self.logger.info("Successfully scraped course stats:")
                self.logger.info(f"Tournament ID: {course_data.get('tournament_id')}")
                self.logger.info(f"Course Name: {course_data.get('course_name')}")
                self.logger.info(f"Course Par: {course_data.get('par')}")
                self.logger.info(f"Course Yardage: {course_data.get('yardage')}")
                
                rounds = course_data.get('rounds', [])
                self.logger.info(f"Rounds found: {len(rounds)}")
                
                for i, round_data in enumerate(rounds):
                    self.logger.info(f"Round {round_data.get('round_number')} - {round_data.get('round_header')}")
                    holes = round_data.get('holes', [])
                    self.logger.info(f"  Holes in round: {len(holes)}")
                    
                    # aboutThisHole to verify sample 
                    if holes and len(holes) > 0:
                        sample_hole = holes[0]
                        self.logger.info(f"  Sample hole {sample_hole.get('hole_number')}:")
                        if 'about_this_hole' in sample_hole:
                            self.logger.info(f"    About this hole: {sample_hole.get('about_this_hole')[:50]}...")
                
                return course_data
            else:
                self.logger.warning("No course data found during test")
                return None
                
        except Exception as e:
            self.logger.error(f"Error test scrape: {str(e)}")
            return None