import pandas as pd
from features import generate_features

def main():
    # Simple procedural flow without Click
    print("Loading raw data...")
    raw = pd.read_csv("data/raw/tennis-master-data.csv")

    print("Generating ELO features...")
    df_feat = generate_features(raw)

    print(f"Features generated! Shape={df_feat.shape}")
    print("✅ Done!")

if __name__ == "__main__":
    main()