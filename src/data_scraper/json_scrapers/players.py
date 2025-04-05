import requests
import json
from ...config import Config

# -m src.data_scraper.json_scrapers.players

# GraphQL query
query = """
query PlayerDirectory($tourCode: TourCode!) {
  playerDirectory(tourCode: $tourCode) {
    players {
      id
      isActive
      firstName
      lastName
      country
    }
  }
}
"""

variables = {
    "tourCode": "R"  # PGA Tour code
}

payload = {
    "query": query,
    "variables": variables
}


response = requests.post(Config.BASE_URL, json=payload, headers=Config.HEADERS)

if response.status_code == 200:
    try:
        data = response.json()
        print("Player Directory Response:", data)  

        # Extract player data if present
        if data and data.get("data") and data["data"].get("playerDirectory"):
            players = data["data"]["playerDirectory"].get("players", [])

            players_data = []
            for player in players:
                player_info = {
                    "id": player.get("id"),
                    "name": f"{player.get('firstName')} {player.get('lastName')}", 
                    "country": player.get("country"),
                    "isActive": player.get("isActive")
                }
                players_data.append(player_info)

            with open("players.json", "w") as json_file:
                json.dump(players_data, json_file, indent=4)
            print("Player data saved to players.json")
        else:
            print("No player data found in the response.")
    except ValueError:
        print("Failed to parse JSON response")
else:
    print(f"Failed to fetch data: {response.status_code}")
    print("Response:", response.text)