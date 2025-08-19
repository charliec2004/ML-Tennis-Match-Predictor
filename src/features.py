import pandas as pd
from elo.elo_update import process_elo

raw_data = pd.read_csv("data/raw/tennis-master-data.csv")


# Adds ELO features, puts feature engineered dataframes in data/processed
def generate_features(raw_data):
    # COPY the data (do not mutate the source data)
    df = raw_data.copy().reset_index(drop=True)

    # prepare lists for new columns
    elo_p1             = []
    elo_p2             = []
    elo_diff           = []

    surf_hard_elo_p1   = []
    surf_grass_elo_p1  = []
    surf_clay_elo_p1   = []
    surf_carpet_elo_p1 = []

    surf_hard_elo_p2   = []
    surf_grass_elo_p2  = []
    surf_clay_elo_p2   = []
    surf_carpet_elo_p2 = []

    # iterate through matches in chronological order
    for _, row in df.iterrows():
        p1 = row['player_1']
        p2 = row['player_2']
        surf = row['surface']  # Capture the surface from CSV
        # CSV’s target: 0 if p1 won, 1 if p2 won
        p1_won = (row['target'] == 0)

        # fetch pre‐match ELOs and update in-memory state
        (elos1, elos2) = process_elo(p1, p2, surf, p1_won)

        # record master ELOs
        elo_p1.append(elos1[0])
        elo_p2.append(elos2[0])
        elo_diff.append(elos1[0] - elos2[0])

        # record surface‐specific ELOs
        surf_hard_elo_p1.append(elos1[1])
        surf_grass_elo_p1.append(elos1[2])
        surf_clay_elo_p1.append(elos1[3])
        surf_carpet_elo_p1.append(elos1[4])

        surf_hard_elo_p2.append(elos2[1])
        surf_grass_elo_p2.append(elos2[2])
        surf_clay_elo_p2.append(elos2[3])
        surf_carpet_elo_p2.append(elos2[4])

    # attach new columns to the DataFrame
    # instead of assigning at the end, insert all new columns before 'target'
    # find the index of the 'target' column
    target_idx = df.columns.get_loc('target')

    # list of (column_name, data_list) in the order you want them inserted
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
    ]

    for name, data in new_cols:
        df.insert(target_idx, name, data)
        target_idx += 1

    # persist enriched dataset and return
    df.to_csv("data/processed/with_elo.csv", index=False)
    return df