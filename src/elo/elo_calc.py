from typing import Tuple

SCALE_CONSTANT = 400.0
DEFAULT_K = 32.0


def _win_probability(p1_elo: float, p2_elo: float, scale: float = SCALE_CONSTANT) -> Tuple[float, float]:
    p1_probability = 1 / (1 + 10 ** ((p2_elo - p1_elo) / scale))
    p2_probability = 1 - p1_probability
    return p1_probability, p2_probability


def calc_elo(p1_elo: float, p2_elo: float, p1_won: bool, k: float = DEFAULT_K) -> Tuple[float, float]:
    p1_prob, p2_prob = _win_probability(p1_elo, p2_elo)

    p1_score = 1.0 if p1_won else 0.0
    p2_score = 1.0 - p1_score

    new_p1_elo = p1_elo + k * (p1_score - p1_prob)
    new_p2_elo = p2_elo + k * (p2_score - p2_prob)

    return new_p1_elo, new_p2_elo
