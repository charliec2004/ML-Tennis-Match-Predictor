import pandas as pd
from elo.elo_processor import EloProcessor
from matches.match_history_processor import MatchHistoryProcessor

def generate_features(raw_data):
    """Generate ELO and match history features using class-based processors"""
    # COPY the data (do not mutate the source data)
    df = raw_data.copy().reset_index(drop=True)

    # Initialize processors
    elo_processor = EloProcessor()
    history_processor = MatchHistoryProcessor()

    # Storage for ELO features (existing code)
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

    # Storage for match history features (NEW)
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

    # iterate through matches in chronological order
    for _, row in df.iterrows():
        p1 = row['player_1']
        p2 = row['player_2']
        surf = row['surface']
        
        # CONVERT STRING DATE TO DATE OBJECT
        match_date = pd.to_datetime(row['date']).date()  # Convert "2000-02-07" -> date(2000, 2, 7)
        
        p1_won = (row['target'] == 0)

        # GET FEATURES BEFORE UPDATING STATE
        
        # Get ELO features (existing code)
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

        # Get match history features (NEW)
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

        # UPDATE STATE AFTER EXTRACTING FEATURES
        elo_processor.update_ratings(p1, p2, surf, p1_won)
        history_processor.update_match_history(p1, p2, match_date, p1_won)  # Now uses date object

    # Add all features to DataFrame (existing + new)
    target_idx = df.columns.get_loc('target')
    
    new_cols = [
        # ELO features (existing)
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
        
        # Match history features (NEW)
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

    # persist enriched dataset and return
    df.to_csv("data/processed/with_features.csv", index=False)
    return df