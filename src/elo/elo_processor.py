import os
import pandas as pd
from typing import Dict, Tuple, Optional
from .elo_calc import calc_elo

class EloProcessor:
    """Manages ELO ratings for all players across all surfaces"""
    
    def __init__(self, players_db_path: Optional[str] = None):
        if players_db_path is None:
            players_db_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir,
                'data', 'raw', 'players_db.csv'
            ))
        
        self.master_elo: Dict[str, float] = {}
        self.surf_hard_elo: Dict[str, float] = {}
        self.surf_grass_elo: Dict[str, float] = {}
        self.surf_clay_elo: Dict[str, float] = {}
        self.surf_carpet_elo: Dict[str, float] = {}
        
        self._load_players(players_db_path)
    
    def _load_players(self, csv_path: str):
        """Initialize all ELO dictionaries with players from CSV"""
        df = pd.read_csv(csv_path, usecols=[0])
        
        names = (
            df['player_name']
              .dropna()
              .str.strip()
              .str.rstrip(',')
        )
        
        self.master_elo = {n: 1500.0 for n in names}
        self.surf_hard_elo = self.master_elo.copy()
        self.surf_grass_elo = self.master_elo.copy()
        self.surf_clay_elo = self.master_elo.copy()
        self.surf_carpet_elo = self.master_elo.copy()
    
    def _case_insensitive_lookup(self, dictionary: Dict[str, float], key: str) -> Tuple[str, bool]:
        """Try to find key in dictionary, first exact match, then case-insensitive"""
        if key in dictionary:
            return key, True
        
        key_lower = key.lower()
        for dict_key in dictionary:
            if dict_key.lower() == key_lower:
                return dict_key, True
        
        return key, False
    
    def _get_player_rating(self, rating_dict: Dict[str, float], player_name: str) -> float:
        """Get rating from dictionary with case-insensitive fallback"""
        actual_key, found = self._case_insensitive_lookup(rating_dict, player_name)
        if found:
            return rating_dict[actual_key]
        else:
            raise KeyError(f"No rating found for player '{player_name}'")
    
    def _update_player_rating(self, rating_dict: Dict[str, float], player_name: str, new_rating: float):
        """Update rating in dictionary with case-insensitive fallback"""
        actual_key, found = self._case_insensitive_lookup(rating_dict, player_name)
        if found:
            rating_dict[actual_key] = new_rating
        else:
            raise KeyError(f"No rating found for player '{player_name}'")
    
    def get_player_ratings(self, player_name: str) -> Tuple[float, float, float, float, float]:
        """Get all ratings for a player: (master, hard, grass, clay, carpet)"""
        return (
            self._get_player_rating(self.master_elo, player_name),
            self._get_player_rating(self.surf_hard_elo, player_name),
            self._get_player_rating(self.surf_grass_elo, player_name),
            self._get_player_rating(self.surf_clay_elo, player_name),
            self._get_player_rating(self.surf_carpet_elo, player_name),
        )
    
    def get_match_features(self, player1: str, player2: str, surface: str) -> Tuple[Tuple[float, float, float, float, float], Tuple[float, float, float, float, float]]:
        """Get current ratings for both players before match (for features)"""
        p1_ratings = self.get_player_ratings(player1)
        p2_ratings = self.get_player_ratings(player2)
        return p1_ratings, p2_ratings
    
    def update_ratings(self, player1: str, player2: str, surface: str, p1_won: bool):
        """Update ratings after a match"""
        p1_ratings = self.get_player_ratings(player1)
        p2_ratings = self.get_player_ratings(player2)
        
        new_master1, new_master2 = calc_elo(p1_ratings[0], p2_ratings[0], p1_won)
        self._update_player_rating(self.master_elo, player1, new_master1)
        self._update_player_rating(self.master_elo, player2, new_master2)
        
        surface_lower = surface.lower()
        if surface_lower == "hard":
            new_surf1, new_surf2 = calc_elo(p1_ratings[1], p2_ratings[1], p1_won)
            self._update_player_rating(self.surf_hard_elo, player1, new_surf1)
            self._update_player_rating(self.surf_hard_elo, player2, new_surf2)
        elif surface_lower == "grass":
            new_surf1, new_surf2 = calc_elo(p1_ratings[2], p2_ratings[2], p1_won)
            self._update_player_rating(self.surf_grass_elo, player1, new_surf1)
            self._update_player_rating(self.surf_grass_elo, player2, new_surf2)
        elif surface_lower == "clay":
            new_surf1, new_surf2 = calc_elo(p1_ratings[3], p2_ratings[3], p1_won)
            self._update_player_rating(self.surf_clay_elo, player1, new_surf1)
            self._update_player_rating(self.surf_clay_elo, player2, new_surf2)
        elif surface_lower == "carpet":
            new_surf1, new_surf2 = calc_elo(p1_ratings[4], p2_ratings[4], p1_won)
            self._update_player_rating(self.surf_carpet_elo, player1, new_surf1)
            self._update_player_rating(self.surf_carpet_elo, player2, new_surf2)
