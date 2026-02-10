# AlphaGolf: Reinforcement Learning for Tournament Outcome Prediction in Professional Golf

**4th Year Computer Science Final Year Project — University of Galway**

AlphaGolf applies Deep Q-Learning to predict PGA Tour tournament winners, built on a custom data pipeline that reverse-engineered the PGA Tour's undocumented GraphQL API to collect comprehensive golf statistics.

## Overview

Professional golf analytics suffers from a lack of publicly available, comprehensive datasets. This project addresses that gap by building a full end-to-end system: from data collection through feature engineering to a reinforcement learning model that frames winner prediction as a sequential decision-making problem.

The system scraped and structured **113.75 MB** of golf data across **11 MongoDB collections**, covering player statistics, tournament histories, scorecards, shot-level data, course characteristics, and weather conditions from 2007–2025.

## Architecture

### Data Pipeline
- **Web Scraping**: Reverse-engineered PGA Tour's GraphQL API (`orchestrator.pgatour.com`) by inspecting network traffic and reconstructing queries without documentation or schema introspection
- **Async Collection**: Built on Python `asyncio` + `aiohttp` for concurrent scraping with rate limiting
- **Data Validation**: Pydantic models enforce schema consistency before database insertion
- **Weather Integration**: Visual Crossing API provides round-by-round meteorological data
- **Storage**: MongoDB for flexible, hierarchical document storage matching the nested API response structure

### Data Collections
| Collection | Coverage | Description |
|---|---|---|
| `player_stats` | 2014–2025 | 132 statistics per player per season |
| `tournament_history` | 2007–2025 | Full leaderboards with round-by-round scoring |
| `scorecards` | 2014–2025 | Hole-by-hole performance data |
| `shot_data` | 2024–2025 | Shot-level coordinate tracking |
| `course_stats` | 2023–2025 | Hole-by-hole course characteristics |
| `current_form` | 2024–2025 | Last 5 tournaments stroke gained metrics |
| `field_stats` | 2025 | Player-course fit scores |
| `player_career` | 2014–2025 | Career statistics and season breakdowns |
| `player_career_profile` | 2014–2025 | OWGR and FedEx Cup rankings |
| `tournament_history_stats` | 2024–2025 | Historical course-specific performance |
| `tournament_weather` | 2022–2025 | Round-by-round weather conditions |

### Feature Engineering
A multi-stage pipeline transforms raw data into ML-ready features:

1. **Extraction**: Modular processors flatten nested MongoDB documents into tabular format
2. **Feature Generation**: Base features (player stats, course, weather), interaction features (player-course, weather adaptation), and temporal features (form trends, momentum)
3. **Analysis & Selection**: Variance filtering, correlation analysis, F-tests, mutual information, and tree-based importance ranking
4. **Winner-Specific Refinement**: Gradient boosting-based feature ranking optimised for the rare-event prediction task

Started with **1,379 candidate features** → refined to **92 winner-optimised features** across 24,000+ player-tournament samples.

### Deep Q-Learning Model
- **Architecture**: 3-layer feedforward network (128 → 64 → 32 neurons, ReLU, dropout 0.2)
- **RL Framing**: Each tournament is an episode; the agent observes player features sequentially and decides to select or skip
- **Reward System**: +10 correct winner, −5 incorrect selection, −1 skipping winner, −2 indecision
- **Training**: Epsilon-greedy exploration (ε: 1.0 → 0.05, decay 0.995), experience replay buffer (5,000)
- **Output**: Softmax-normalised win probabilities per tournament field

## Tech Stack

- **Language**: Python
- **Scraping**: `aiohttp`, `asyncio`, custom GraphQL query templates
- **Validation**: Pydantic
- **Database**: MongoDB
- **ML/DL**: TensorFlow, scikit-learn
- **Data Processing**: pandas, NumPy
- **Weather API**: Visual Crossing and PGA Tour's undocumented API

## Setup

```bash
# Clone the repo
git clone https://github.com/johnstew324/AlphaGolf-FYP.git
cd AlphaGolf-FYP

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Add your API keys: PGA_API_KEY, MONGODB_URI, VISUAL_CROSSING_API_KEY
```

## Project Structure

```
AlphaGolf-FYP/
├── src/
│   ├── scrapers/          # Async scraper modules (player, tournament, weather)
│   ├── processors/        # Data extraction & flattening from MongoDB
│   ├── features/          # Feature generation, analysis, selection, transformation
│   ├── model/             # DQN architecture, training loop, evaluation
│   └── utils/             # Config, database connections, shared utilities
├── alphagolf/
│   └── raw_jsons/         # Bootstrap metadata (player/tournament directories)
├── requirements.txt

## Acknowledgements

- **Supervisor**: Effirul Ramlan, University of Galway
- **Data Sources**: [PGA Tour](https://www.pgatour.com/), [Visual Crossing Weather API](https://www.visualcrossing.com/)
- **References**: See full report for complete bibliography