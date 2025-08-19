import os, pandas as pd
from typing import Dict

# build absolute path to the CSV (up two levels to project root)
CSV_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    'data', 'raw', 'players_db.csv'
))

df = pd.read_csv(CSV_PATH, usecols=[0])



# placeholders
master_elo: Dict[str, float]      = {}
surf_hard_elo: Dict[str, float]   = {}
surf_grass_elo: Dict[str, float]  = {}
surf_clay_elo: Dict[str, float]   = {}
surf_carpet_elo: Dict[str, float] = {}


# Build the elo state dicts
def init_elo_state():
    global master_elo, surf_hard_elo, surf_grass_elo, surf_clay_elo, surf_carpet_elo

    # clean up any trailing commas & whitespace on each name
    names = (
        df['player_name']
          .dropna()
          .str.strip()       # trim whitespace
          .str.rstrip(',')   # remove trailing commas
    )

    master_elo      = {n: 1500.0 for n in names}
    surf_hard_elo   = master_elo.copy()
    surf_grass_elo  = master_elo.copy()
    surf_clay_elo   = master_elo.copy()
    surf_carpet_elo = master_elo.copy()



# Add a helper function to handle case-insensitive lookups
def _case_insensitive_lookup(dictionary, key):
    """
    Try to find key in dictionary, first with exact match, 
    then case-insensitive. Returns (found_key, found) tuple.
    """
    # Try exact match first
    if key in dictionary:
        return key, True
    
    # Try case-insensitive match
    key_lower = key.lower()
    for dict_key in dictionary:
        if dict_key.lower() == key_lower:
            print(f"Case mismatch: '{key}' found as '{dict_key}'")
            return dict_key, True
    
    # Not found at all
    return key, False

# ---- GET FUNCTIONS ----

def get_player_master_elo(player_name: str) -> float:
    actual_key, found = _case_insensitive_lookup(master_elo, player_name)
    if found:
        return master_elo[actual_key]
    else:
        raise KeyError(f"No master ELO found for player '{player_name}'")

def get_player_surf_hard_elo(player_name: str) -> float:
    actual_key, found = _case_insensitive_lookup(surf_hard_elo, player_name)
    if found:
        return surf_hard_elo[actual_key]
    else:
        raise KeyError(f"No hard ELO found for player '{player_name}'")

def get_player_surf_grass_elo(player_name: str) -> float:
    actual_key, found = _case_insensitive_lookup(surf_grass_elo, player_name)
    if found:
        return surf_grass_elo[actual_key]
    else:
        raise KeyError(f"No grass ELO found for player '{player_name}'")

def get_player_surf_clay_elo(player_name: str) -> float:
    actual_key, found = _case_insensitive_lookup(surf_clay_elo, player_name)
    if found:
        return surf_clay_elo[actual_key]
    else:
        raise KeyError(f"No clay ELO found for player '{player_name}'")

def get_player_surf_carpet_elo(player_name: str) -> float:
    actual_key, found = _case_insensitive_lookup(surf_carpet_elo, player_name)
    if found:
        return surf_carpet_elo[actual_key]
    else:
        raise KeyError(f"No carpet ELO found for player '{player_name}'")

# ---- UPDATE FUNCTIONS ----
def update_player_master_elo(player_name: str, new_elo: float) -> None:
    actual_key, found = _case_insensitive_lookup(master_elo, player_name)
    if found:
        master_elo[actual_key] = new_elo
    else:
        raise KeyError(f"No master ELO found for player '{player_name}'")

def update_player_surf_hard_elo(player_name: str, new_elo: float) -> None:
    actual_key, found = _case_insensitive_lookup(surf_hard_elo, player_name)
    if found:
        surf_hard_elo[actual_key] = new_elo
    else:
        raise KeyError(f"No hard ELO found for player '{player_name}'")

def update_player_surf_grass_elo(player_name: str, new_elo: float) -> None:
    actual_key, found = _case_insensitive_lookup(surf_grass_elo, player_name)
    if found:
        surf_grass_elo[actual_key] = new_elo
    else:
        raise KeyError(f"No grass ELO found for player '{player_name}'")

def update_player_surf_clay_elo(player_name: str, new_elo: float) -> None:
    actual_key, found = _case_insensitive_lookup(surf_clay_elo, player_name)
    if found:
        surf_clay_elo[actual_key] = new_elo
    else:
        raise KeyError(f"No clay ELO found for player '{player_name}'")

def update_player_surf_carpet_elo(player_name: str, new_elo: float) -> None:
    actual_key, found = _case_insensitive_lookup(surf_carpet_elo, player_name)
    if found:
        surf_carpet_elo[actual_key] = new_elo
    else:
        raise KeyError(f"No carpet ELO found for player '{player_name}'")



# VERY IMPORTANT!!!
# ACTUALLY CREATES THE ELO STATE DICTS
init_elo_state()

# if __name__ == '__main__':
#     for player, rating in master_elo.items():
#         print(f"{player}: {rating}")
#     print("Any missing names?", df['player_name'].isna().any())
#     print("Missing names count:", df['player_name'].isna().sum())
#     print("Total players:", len(master_elo))
    # print(get_player_master_elo('Amritraj P.'))
