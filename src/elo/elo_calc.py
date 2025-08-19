# Functions used for calclulating ELO

# For clearly defined typing
from typing import Tuple

"""
NOTE:
ELO updates work like this:
- Start with the players' previous ELOs.
- Compute the expected win probability for each (based on rating difference).
- Compare expected result with actual result.
- Update each player's ELO: 
    new_rating = old_rating + K * (actual_score - expected_score)

Where:
- actual_score = 1 if player won, 0 if lost
- expected_score = probability of winning from the ELO formula
- K is the learning rate (controls how fast ratings move)

So the expected probability **scales** the size of the ELO change.
"""

# --------- CONSTANTS ---------

SCALE_CONSTANT = 400.0;
DEFAULT_K = 32.0

'''
TODO: Possibly make K dynamic:
        - Early-career players: uncertain → K *= (1 + max(0, 20 - matches_played)/20). Caps at 2x for debut, fades to 1x by 20 matches.
        - Best of 5 (increasing K)
        - LOWER K for surface ELO's maybe ???
'''


# --------- HELPER FUNCTIONS ---------

def _win_probability(p1_elo: float, p2_elo: float, scale: float = SCALE_CONSTANT) -> Tuple[float, float]:
    p1_probability = 1/(1+10 ** ((p2_elo - p1_elo)/scale))
    p2_probaility = 1-p1_probability
    return p1_probability, p2_probaility


# --------- CALCULATE ELO MASTER FUNCTION ---------

def calc_elo(p1_elo: float, p2_elo: float, p1_won: bool, k: float = DEFAULT_K) -> Tuple[float, float]:
    
    # Calculate the probabilty of each to win based on elo (determines if a win is an upset or expected)
    p1_prob , p2_prob = _win_probability(p1_elo, p2_elo)

    # Determine the winner of the match
    p1_score: float = 1.0 if p1_won else 0.0
    p2_score: float = 1.0 - p1_score

    # Calculate new ELOs
    new_p1_elo: float = p1_elo + k * (p1_score - p1_prob)
    new_p2_elo: float = p2_elo + k * (p2_score - p2_prob)

    # Return a tuple of both elos
    return new_p1_elo , new_p2_elo