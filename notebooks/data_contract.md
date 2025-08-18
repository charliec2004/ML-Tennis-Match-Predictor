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

- elo_A_pre
- elo_B_pre
- elo_diff
- surf_elo_A_pre
- surf_elo_B_pre
- surf_elo_diff
- rest_days_A
- rest_days_B
- h2h_A
- h2h_B
- recent_wr_A
- recent_wr_B

### Target

- target (1 if player_1 wins, 0 if not)
