import requests
import json
from ...config import Config

 # "python -m src.data_scraper.json_scrapers.locations_dates"

def fetch_schedule_for_year(year):
    payload = {
        "operationName": "Schedule",
        "query": """query Schedule($tourCode: String!, $year: String, $filter: TournamentCategory) {
          schedule(tourCode: $tourCode, year: $year, filter: $filter) {
            completed {
              month
              year
              monthSort
              tournaments {
                tournamentName
                id
                city
                country
                courseName
                date
                startDate
                state
                status {
                  roundDisplay
                  roundStatus
                }
                tournamentStatus
              }
            }
            upcoming {
              month
              year
              monthSort
              tournaments {
                tournamentName
                id
                city
                country
                courseName
                date
                startDate
                state
                status {
                  roundDisplay
                  roundStatus
                }
                tournamentStatus
              }
            }
          }
        }""",
        "variables": {
            "tourCode": "R",
            "year": str(year)
        }
    }

    extracted_data = {
        "Year": year,
        "Completed Tournaments": [],
        "Upcoming Tournaments": []
    }

    try:
        response = requests.post(Config.BASE_URL, json=payload, headers=Config.HEADERS)
        print(f"Status Code for {year}: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            schedule = data.get("data", {}).get("schedule", {})

            for tournament in schedule.get("completed", []):
                for t in tournament.get("tournaments", []):
                    status = t.get("status") or {}
                    tournament_info = {
                        "Tournament Name": t.get("tournamentName"),
                        "Tournament ID": t.get("id"),
                        "City": t.get("city"),
                        "State": t.get("state"),
                        "Country": t.get("country"),
                        "Course Name": t.get("courseName"),
                        "Date": t.get("date"),
                        "Start Date": t.get("startDate"),
                        "Tournament Status": t.get("tournamentStatus")
                    }
                    if status.get("roundDisplay"):
                        tournament_info["Round Display"] = status.get("roundDisplay")
                    if status.get("roundStatus"):
                        tournament_info["Round Status"] = status.get("roundStatus")
                    extracted_data["Completed Tournaments"].append(tournament_info)

            for tournament in schedule.get("upcoming", []):
                for t in tournament.get("tournaments", []):
                    status = t.get("status") or {}
                    tournament_info = {
                        "Tournament Name": t.get("tournamentName"),
                        "Tournament ID": t.get("id"),
                        "City": t.get("city"),
                        "State": t.get("state"),
                        "Country": t.get("country"),
                        "Course Name": t.get("courseName"),
                        "Date": t.get("date"),
                        "Start Date": t.get("startDate"),
                        "Tournament Status": t.get("tournamentStatus")
                    }
                    if status.get("roundDisplay"):
                        tournament_info["Round Display"] = status.get("roundDisplay")
                    if status.get("roundStatus"):
                        tournament_info["Round Status"] = status.get("roundStatus")
                    extracted_data["Upcoming Tournaments"].append(tournament_info)

        else:
            print(f"Error for {year}: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {year}: {e}")
    
    return extracted_data


years_to_fetch = [2022, 2023, 2024, 2025]  # Fetching data from 2019 to 2025

all_tournaments = []
for year in years_to_fetch:
    year_data = fetch_schedule_for_year(year)
    all_tournaments.append(year_data)


file_path = r"C:\Users\johns\AlphaGOLF-FYP\AlphaGolf\data\raw_jsons\tournaments_location_dates.json"

# Save the data to the specified file path
with open(file_path, 'w') as file:
    json.dump(all_tournaments, file, indent=4)

print(f"Data saved to {file_path}")


# rememeber to run using "pyhton -m src.data_scraper.json_scrapers.locations_dates"