import os
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Tuple, Optional

class MatchHistoryProcessor:
    """Manages match history for all players to compute win rates and H2H records"""
    ''' Core stats: 
            - Individual win rate in the last x matches
            - All time H2H win rate
            - Days Since last match
    '''
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
        
        # Clean up player names
        names = (
            df['player_name']
              .dropna()
              .str.strip()       # trim whitespace
              .str.rstrip(',')   # remove trailing commas
        )
        
        # Initialize empty match history for each player
        self.match_history = {n: [] for n in names}
    
    def _case_insensitive_lookup(self, key: str) -> Tuple[str, bool]:
        """Try to find key in match_history, first exact match, then case-insensitive"""
        # Try exact match first
        if key in self.match_history:
            return key, True
        
        # Try case-insensitive match
        key_lower = key.lower()
        for dict_key in self.match_history:
            if dict_key.lower() == key_lower:
                # print(f"Case mismatch: '{key}' found as '{dict_key}'")
                return dict_key, True
        
        # Not found at all
        return key, False
    
    def get_win_rate(self, player: str, amt_of_matches: int = 0) -> float:
        """GET a player's win rate in the last X matches"""
        
        # Handle case-insensitive player lookup
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return 0.5  # Default for unknown players
        
        matches = self.match_history[actual_key]
        if not matches:
            return 0.5  # Default for players with no match history
        
        if amt_of_matches == 0:  # If 0, we are looking for ALL matches
            relevant_matches = matches
        else:
            # Take the most recent amt_of_matches (matches are chronological)
            relevant_matches = matches[-amt_of_matches:]
        
        if not relevant_matches:
            return 0.5
        
        # Count wins (did_lose = False means won)
        # Remember: tuple structure is (opponent_name, match_date, did_lose)
        # GENERATOR EXPRESSION TO SAVE MEMORY
        wins = sum(1 for _, _, did_lose in relevant_matches if not did_lose)
        
        return wins / len(relevant_matches)
    
    def get_h2h_win_rate(self, player1: str, player2: str, amt_of_matches: int = 0) -> float:
        """GET a players H2H win rate AGAINST another player in the last X matches
    
        Returns % of times player1 has won only against player2
        NOTE: If the two players do not have any previous matches then, DEFAULT to 0.5
        """
    
        # Handle case-insensitive lookups for both players
        actual_key1, found1 = self._case_insensitive_lookup(player1)
        actual_key2, found2 = self._case_insensitive_lookup(player2)
    
        if not found1 or not found2:
            return 0.5  # Default for unknown players
    
        # Get all H2H matches where player1 played against player2
        # tuple structure: (opponent_name, match_date, did_lose)
        h2h_matches = [
            (opponent, match_date, did_lose) for opponent, match_date, did_lose in self.match_history[actual_key1]
            if opponent.lower() == actual_key2.lower()  # Case-insensitive opponent comparison
        ]
    
        if not h2h_matches:
            return 0.5  # No previous H2H matches
    
        if amt_of_matches == 0:  # All H2H matches
            relevant_h2h = h2h_matches
        else:
            # Take most recent H2H matches
            relevant_h2h = h2h_matches[-amt_of_matches:]
    
        if not relevant_h2h:
            return 0.5
    
        # Count wins in H2H (did_lose = False means player1 won)
        h2h_wins = sum(1 for _, _, did_lose in relevant_h2h if not did_lose)
    
        return h2h_wins / len(relevant_h2h)
    
    def get_total_matches_played(self, player: str) -> int:
        """GET total matches played, useful for determining whether or not a player has enough matches"""
        
        # Handle case-insensitive player lookup
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return 0  # Unknown player has 0 matches
        
        return len(self.match_history[actual_key])
    
    def days_since_last_match(self, player: str, current_date: date) -> float:
        """GET the last match played's DATE, subtract it from the current date"""
        
        # Handle case-insensitive player lookup
        actual_key, found = self._case_insensitive_lookup(player)
        
        if not found:
            return np.nan  # Very large number for unknown players (indicates "long time ago")
        
        matches = self.match_history[actual_key]
        if not matches:
            return np.nan  # Very large number for players with no matches
        
        # Get the most recent match (matches are in chronological order)
        # tuple structure: (opponent_name, match_date, did_lose)
        last_match_date = matches[-1][1]  # Get the date from the last match
        
        # Calculate days difference
        days_diff = (current_date - last_match_date).days
        
        return max(0, days_diff)  # Ensure non-negative (in case of date issues)
    
    def update_match_history(self, player1: str, player2: str, match_date: date, p1_won: bool):
        """ADD a new match to both players' match history
    
        Should be done AFTER extracting features to prevent data leakage
        """
    
        # Handle case-insensitive lookups
        actual_key1, found1 = self._case_insensitive_lookup(player1)
        actual_key2, found2 = self._case_insensitive_lookup(player2)
    
        # Add match to player1's history
        if found1:
            # For player1: did_lose = not p1_won
            self.match_history[actual_key1].append((actual_key2, match_date, not p1_won))
        else:
            print(f"Warning: Player '{player1}' not found in match history database")
    
        # Add match to player2's history  
        if found2:
            # For player2: did_lose = p1_won (opposite of player1)
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
    
        # NOTE: DO NOT call update_match_history here (seperation of concerns)