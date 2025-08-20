import pandas as pd
from features import generate_features

''' Main Function '''
def main():
    # Simple procedural flow without Click
    print("Loading raw data...")
    raw = pd.read_csv("data/raw/tennis-master-data.csv")

    # Generated ELO and match history stats
    print("Generating features...")
    df_processed = generate_features(raw)

    print(f"Features generated! Shape={df_processed.shape}")

if __name__ == "__main__":
    main()