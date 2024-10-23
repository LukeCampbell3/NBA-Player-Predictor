import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from datetime import timedelta
import os
import time  # Import time module for sleep function

def format_player_code(full_name):
    names = full_name.strip().split()
    first_name = names[0]
    last_name = names[-1]
    player_code = last_name[:5].lower() + first_name[:2].lower() + '01'
    return player_code

def get_player_position(soup):
    position_section = soup.find_all('p')
    for p in position_section:
        if 'Position:' in p.text:
            position = p.text.split('Position:')[1].split('•')[0].strip()
            return position.split(' ')[0]
    return 'N/A'  # Default if not found

def get_defensive_rating(opponent_abbr, year):
    url = f"https://www.basketball-reference.com/teams/{opponent_abbr}/{year}.html"
    response = requests.get(url)
    if response.status_code != 200:
        return 'N/A'  # Return a placeholder if unable to fetch the data
    soup = BeautifulSoup(response.content, 'html.parser')
    drtg_cell = soup.find('td', {'data-stat': 'def_rtg'})
    if drtg_cell:
        return drtg_cell.text
    return 'N/A'

def scrape_player_game_stats(full_name, year):
    player_code = format_player_code(full_name)
    first_letter = player_code[0]
    basic_url = f"https://www.basketball-reference.com/players/{first_letter}/{player_code}/gamelog/{year}"
    advanced_url = f"https://www.basketball-reference.com/players/{first_letter}/{player_code}/gamelog-advanced/{year}"
    
    basic_response = requests.get(basic_url)
    advanced_response = requests.get(advanced_url)
    if basic_response.status_code != 200 or advanced_response.status_code != 200:
        return []

    basic_soup = BeautifulSoup(basic_response.content, 'html.parser')
    advanced_soup = BeautifulSoup(advanced_response.content, 'html.parser')
    position = get_player_position(basic_soup)
    basic_table = basic_soup.find('table', {'id': 'pgl_basic'})
    advanced_table = advanced_soup.find('table', {'id': 'pgl_advanced'})

    if not basic_table or not advanced_table:
        return []

    player_stats = []

    basic_rows = basic_table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')
    advanced_rows = advanced_table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')

    for basic_row, advanced_row in zip(basic_rows, advanced_rows):
        opponent_abbr = basic_row.find('td', {'data-stat': 'opp_id'}).text.strip() if basic_row.find('td', {'data-stat': 'opp_id'}) else 'N/A'
        if opponent_abbr != 'N/A':
            team_mappings = {
                'CHO' : 'CHA',
                'NOH' : 'NOP',
                'NJN' : 'BKN', 
                }
            opponent_abbr = team_mappings.get(opponent_abbr, opponent_abbr)
            opp_def_rating = get_defensive_rating(opponent_abbr, year)
        else:
            opp_def_rating = 'N/A'

        game_date_str = basic_row.find('td', {'data-stat': 'date_game'}).text.strip()
        game_stats = {
            'Player' : full_name,
            'Position' : position,
            'Date': game_date_str,
            # Add traditional stats
            'TM' : basic_row.find('td', {'data-stat': 'team_id'}).text.strip() if basic_row.find('td', {'data-stat': 'team_id'}) else 'N/A',
            'OPP': advanced_row.find('td', {'data-stat': 'opp_id'}).text.strip() if advanced_row.find('td', {'data-stat': 'opp_id'}) else 'N/A',
            'PTS': basic_row.find('td', {'data-stat': 'pts'}).text.strip() if basic_row.find('td', {'data-stat': 'pts'}) else 'N/A',

            'DRTG' : opp_def_rating
        }
            
        player_stats.append(game_stats)

        time.sleep(10)  # Delay added here

    return player_stats

# Example usage
full_name = "James Harden"
year = "2024"
stats = scrape_player_game_stats(full_name, year)
df = pd.DataFrame(stats)
folder_name = 'Data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
csv_file = os.path.join(folder_name, f"{full_name}_{year}.csv")
df.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")
