from typing import Tuple
from elo.elo_state import (
    get_player_master_elo,
    get_player_surf_hard_elo,
    get_player_surf_grass_elo,
    get_player_surf_clay_elo,
    get_player_surf_carpet_elo,
    update_player_master_elo,
    update_player_surf_hard_elo,
    update_player_surf_grass_elo,
    update_player_surf_clay_elo,
    update_player_surf_carpet_elo,
)
from elo.elo_calc import calc_elo


def process_elo(
    player1_name: str,
    player2_name: str,
    p1_won: bool
) -> Tuple[
    Tuple[float, float, float, float, float],
    Tuple[float, float, float, float, float]
]:
    """
    1) Fetch pre‐match master & surface ELOs for both players  
    2) Compute new ELOs via calc_elo()  
    3) Persist updates into elo_state  
    4) Return the old ratings as two 5‐tuples
    """
    # 1) fetch
    p1_elos = (
        get_player_master_elo(player1_name),
        get_player_surf_hard_elo(player1_name),
        get_player_surf_grass_elo(player1_name),
        get_player_surf_clay_elo(player1_name),
        get_player_surf_carpet_elo(player1_name),
    )
    p2_elos = (
        get_player_master_elo(player2_name),
        get_player_surf_hard_elo(player2_name),
        get_player_surf_grass_elo(player2_name),
        get_player_surf_clay_elo(player2_name),
        get_player_surf_carpet_elo(player2_name),
    )

    # 2) compute
    new_master1, new_master2 = calc_elo(p1_elos[0], p2_elos[0], p1_won)
    new_hard1,   new_hard2   = calc_elo(p1_elos[1], p2_elos[1], p1_won)
    new_grass1,  new_grass2  = calc_elo(p1_elos[2], p2_elos[2], p1_won)
    new_clay1,   new_clay2   = calc_elo(p1_elos[3], p2_elos[3], p1_won)
    new_carpet1, new_carpet2 = calc_elo(p1_elos[4], p2_elos[4], p1_won)

    # 3) persist
    update_player_master_elo(player1_name, new_master1)
    update_player_master_elo(player2_name, new_master2)
    update_player_surf_hard_elo(player1_name, new_hard1)
    update_player_surf_hard_elo(player2_name, new_hard2)
    update_player_surf_grass_elo(player1_name, new_grass1)
    update_player_surf_grass_elo(player2_name, new_grass2)
    update_player_surf_clay_elo(player1_name, new_clay1)
    update_player_surf_clay_elo(player2_name, new_clay2)
    update_player_surf_carpet_elo(player1_name, new_carpet1)
    update_player_surf_carpet_elo(player2_name, new_carpet2)

    # 4) return old ELOs
    return p1_elos, p2_elos