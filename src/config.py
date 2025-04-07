import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_URL = "https://orchestrator.pgatour.com/graphql"
    HEADERS = {
        "Content-Type": "application/json",
        "Referer": "https://www.pgatour.com/",
        "Origin": "https://www.pgatour.com/",
        "X-Amz-User-Agent": "aws-amplify/3.0.7",
        "X-Api-Key": os.getenv('PGA_API_KEY', ''),
        "X-Pgat-Platform": "web"
    }
    MONGODB_URI = os.getenv('MONGODB_URI')
    
    # weather API Configuration
    VISUAL_CROSSING_API_KEY = os.getenv('VISUAL_CROSSING_API_KEY', '')
    
       
    ## '' is the default value if the key is not found in the .env file