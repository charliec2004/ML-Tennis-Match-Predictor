import os
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Tuple, Optional

class MatchHistoryProcessor:
    """Manages match history for all players to compute win rates and H2H records"""
    def __init__(self, players_db_path: Optional[str] = None):
        if players_db_path is None:
            players_db_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir,
                'data', 'raw', 'players_db.csv'
            ))
        
        self.match_history: Dict[str, List[Tuple[str, date ,bool]]] = {}
        self._load_players(players_db_path)
    
    def _load_players(self, csv_path: str):
        """Initialize empty match history for all players"""
        df = pd.read_csv(csv_path, usecols=[0])
        
        names = (
            df['player_name']
              .dropna()
              .str.strip()
              .str.rstrip(',')
        )
        
        self.match_history = {n: [] for n in names}
    
    def _case_insensitive_lookup(self, key: str) -> Tuple[str, bool]:
        """Try to find key in match_history, first exact match, then case-insensitive"""
        if key in self.match_history:
            return key, True
        
        key_lower = key.lower()
        for dict_key in self.match_history:
            if dict_key.lower() == key_lower:
                return dict_key, True
        
        return key, False
    
    def get_win_rate(self, player: str, amt_of_matches: int = 0) -> float:
        """GET a player's win rate in the last X matches"""
        
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return 0.5
        
        matches = self.match_history[actual_key]
        if not matches:
            return 0.5
        
        if amt_of_matches == 0:
            relevant_matches = matches
        else:
            relevant_matches = matches[-amt_of_matches:]
        
        if not relevant_matches:
            return 0.5
        
        wins = sum(1 for _, _, did_lose in relevant_matches if not did_lose)
        
        return wins / len(relevant_matches)
    
    def get_h2h_win_rate(self, player1: str, player2: str, amt_of_matches: int = 0) -> float:
        """GET a players H2H win rate AGAINST another player in the last X matches
    
        Returns % of times player1 has won only against player2
        NOTE: If the two players do not have any previous matches then, DEFAULT to 0.5
        """
    
        actual_key1, found1 = self._case_insensitive_lookup(player1)
        actual_key2, found2 = self._case_insensitive_lookup(player2)
    
        if not found1 or not found2:
            return 0.5
    
        h2h_matches = [
            (opponent, match_date, did_lose) for opponent, match_date, did_lose in self.match_history[actual_key1]
            if opponent.lower() == actual_key2.lower()
        ]
    
        if not h2h_matches:
            return 0.5
    
        if amt_of_matches == 0:
            relevant_h2h = h2h_matches
        else:
            relevant_h2h = h2h_matches[-amt_of_matches:]
    
        if not relevant_h2h:
            return 0.5
    
        h2h_wins = sum(1 for _, _, did_lose in relevant_h2h if not did_lose)
    
        return h2h_wins / len(relevant_h2h)
    
    def get_total_matches_played(self, player: str) -> int:
        """GET total matches played, useful for determining whether or not a player has enough matches"""
        
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return 0
        
        return len(self.match_history[actual_key])
    
    def days_since_last_match(self, player: str, current_date: date) -> float:
        """GET the last match played's DATE, subtract it from the current date"""
        
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return np.nan
        
        matches = self.match_history[actual_key]
        if not matches:
            return np.nan
        
        last_match_date = matches[-1][1]
        
        days_diff = (current_date - last_match_date).days
        
        return max(0, days_diff)
    
    def update_match_history(self, player1: str, player2: str, match_date: date, p1_won: bool):
        """ADD a new match to both players' match history
    
        Should be done AFTER extracting features to prevent data leakage
        """
    
        actual_key1, found1 = self._case_insensitive_lookup(player1)
        actual_key2, found2 = self._case_insensitive_lookup(player2)
    
        if found1:
            self.match_history[actual_key1].append((actual_key2, match_date, not p1_won))
        else:
            print(f"Warning: Player '{player1}' not found in match history database")
    
        if found2:
            self.match_history[actual_key2].append((actual_key1, match_date, p1_won))
        else:
            print(f"Warning: Player '{player2}' not found in match history database")
    
    def get_match_history_features(self, player1: str, player2: str, current_date: date) -> Dict[str, float]:
        """Get all match history features for both players (for use in features.py)"""
        return {
            'win_rate_all_p1': self.get_win_rate(player1),
            'win_rate_all_p2': self.get_win_rate(player2),
            'win_rate_5_p1': self.get_win_rate(player1, 5),
            'win_rate_5_p2': self.get_win_rate(player2, 5),
            'win_rate_10_p1': self.get_win_rate(player1, 10),
            'win_rate_10_p2': self.get_win_rate(player2, 10),
            'h2h_win_rate_p1': self.get_h2h_win_rate(player1, player2),
            'total_matches_p1': float(self.get_total_matches_played(player1)),
            'total_matches_p2': float(self.get_total_matches_played(player2)),
            'days_since_last_p1': self.days_since_last_match(player1, current_date),
            'days_since_last_p2': self.days_since_last_match(player2, current_date),
        }
