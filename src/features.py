import pandas as pd
from elo.elo_processor import EloProcessor
from matches.match_history_processor import MatchHistoryProcessor

def generate_features(raw_data):
    """Generate ELO and match history features using class-based processors"""
    df = raw_data.copy().reset_index(drop=True)
    
    # Ensure chronological order to prevent data leakage
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    elo_processor = EloProcessor()
    history_processor = MatchHistoryProcessor()

    elo_p1 = []
    elo_p2 = []
    elo_diff = []

    surf_hard_elo_p1   = []
    surf_grass_elo_p1  = []
    surf_clay_elo_p1   = []
    surf_carpet_elo_p1 = []

    surf_hard_elo_p2   = []
    surf_grass_elo_p2  = []
    surf_clay_elo_p2   = []
    surf_carpet_elo_p2 = []

    win_rate_all_p1 = []
    win_rate_all_p2 = []
    win_rate_5_p1 = []
    win_rate_5_p2 = []
    win_rate_10_p1 = []
    win_rate_10_p2 = []
    h2h_win_rate_p1 = []
    total_matches_p1 = []
    total_matches_p2 = []
    days_since_last_p1 = []
    days_since_last_p2 = []

    for _, row in df.iterrows():
        p1 = row['player_1']
        p2 = row['player_2']
        surf = row['surface']

        match_date = pd.to_datetime(row['date']).date()

        p1_won = (row['target'] == 0)

        p1_ratings, p2_ratings = elo_processor.get_match_features(p1, p2, surf)
        elo_p1.append(p1_ratings[0])
        elo_p2.append(p2_ratings[0])
        elo_diff.append(p1_ratings[0] - p2_ratings[0])

        surf_hard_elo_p1.append(p1_ratings[1])
        surf_grass_elo_p1.append(p1_ratings[2])
        surf_clay_elo_p1.append(p1_ratings[3])
        surf_carpet_elo_p1.append(p1_ratings[4])

        surf_hard_elo_p2.append(p2_ratings[1])
        surf_grass_elo_p2.append(p2_ratings[2])
        surf_clay_elo_p2.append(p2_ratings[3])
        surf_carpet_elo_p2.append(p2_ratings[4])

        history_features = history_processor.get_match_history_features(p1, p2, match_date)
        win_rate_all_p1.append(history_features['win_rate_all_p1'])
        win_rate_all_p2.append(history_features['win_rate_all_p2'])
        win_rate_5_p1.append(history_features['win_rate_5_p1'])
        win_rate_5_p2.append(history_features['win_rate_5_p2'])
        win_rate_10_p1.append(history_features['win_rate_10_p1'])
        win_rate_10_p2.append(history_features['win_rate_10_p2'])
        h2h_win_rate_p1.append(history_features['h2h_win_rate_p1'])
        total_matches_p1.append(history_features['total_matches_p1'])
        total_matches_p2.append(history_features['total_matches_p2'])
        days_since_last_p1.append(history_features['days_since_last_p1'])
        days_since_last_p2.append(history_features['days_since_last_p2'])

        elo_processor.update_ratings(p1, p2, surf, p1_won)
        history_processor.update_match_history(p1, p2, match_date, p1_won)

    target_idx = df.columns.get_loc('target')
    
    new_cols = [
        ('elo_p1', elo_p1),
        ('elo_p2', elo_p2),
        ('elo_diff', elo_diff),
        ('surf_hard_elo_p1', surf_hard_elo_p1),
        ('surf_grass_elo_p1', surf_grass_elo_p1),
        ('surf_clay_elo_p1', surf_clay_elo_p1),
        ('surf_carpet_elo_p1', surf_carpet_elo_p1),
        ('surf_hard_elo_p2', surf_hard_elo_p2),
        ('surf_grass_elo_p2', surf_grass_elo_p2),
        ('surf_clay_elo_p2', surf_clay_elo_p2),
        ('surf_carpet_elo_p2', surf_carpet_elo_p2),
        ('win_rate_all_p1', win_rate_all_p1),
        ('win_rate_all_p2', win_rate_all_p2),
        ('win_rate_5_p1', win_rate_5_p1),
        ('win_rate_5_p2', win_rate_5_p2),
        ('win_rate_10_p1', win_rate_10_p1),
        ('win_rate_10_p2', win_rate_10_p2),
        ('h2h_win_rate_p1', h2h_win_rate_p1),
        ('total_matches_p1', total_matches_p1),
        ('total_matches_p2', total_matches_p2),
        ('days_since_last_p1', days_since_last_p1),
        ('days_since_last_p2', days_since_last_p2),
    ]

    for name, data in new_cols:
        df.insert(target_idx, name, data)
        target_idx += 1

    df.to_csv("data/processed/with_features.csv", index=False)
    return df
