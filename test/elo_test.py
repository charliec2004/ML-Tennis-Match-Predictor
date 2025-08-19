import sys, os
# ensure `src/` is on the import path so `elo` can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from elo.elo_calc import update_ELO  # type: ignore

# Demonstrate update_ELO: player1 ELO=1600 vs player2 ELO=1500, player1 wins
# simple demonstration when run directly
if __name__ == "__main__":
    new_p1_elo, new_p2_elo = update_ELO(3000.0, 1500.0, False)
    print(f"New ELOs => Player1: {new_p1_elo}, Player2: {new_p2_elo}")