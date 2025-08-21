# Data Contract

## Columns

### Meta (carry through, not used in training)

- MATCH_ID
- date
- tournament
- player_1
- player_2
- winner
- score

### Static Features (from raw CSV)

- series_level
- best_of_3
- best_of_5
- round
- is_outdoor
- surf_fast
- surf_hard
- surf_clay
- surf_grass
- surf_carpet
- rank_1
- rank_2
- rank_avg
- rank_ratio
- rank_diff
- is_top10_match

### Dynamic Features (built in pipeline)

- elo_p1
- elo_p2
- elo_diff
- surf_hard_elo_p1
- surf_grass_elo_p1
- surf_clay_elo_p1
- surf_carpet_elo_p1
- surf_hard_elo_p2
- surf_grass_elo_p2
- surf_clay_elo_p2
- surf_carpet_elo_p2
- win_rate_all_p1
- win_rate_all_p2
- win_rate_5_p1
- win_rate_5_p2
- win_rate_10_p1
- win_rate_10_p2
- h2h_win_rate_p1
- total_matches_p1
- total_matches_p2
- days_since_last_p1
- days_since_last_p2

### Target

- target (1 if player_1 wins, 0 if not)
