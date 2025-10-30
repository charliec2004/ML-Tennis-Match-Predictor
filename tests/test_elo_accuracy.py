import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))


def test_elo_accuracy():
    """
    Test to see what percentage of the time the higher ELO player wins.
    This validates that our ELO system has predictive power.
    """
    
    print("Testing ELO Accuracy: How often does the higher ELO player win?")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'with_features.csv'
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} matches from {data_path}")
    except FileNotFoundError:
        print(f"Could not find {data_path}")
        print("   Please run your feature generation first to create the CSV with ELO ratings.")
        return
    
    df_test = df.iloc[100:].copy()
    print(f"Testing on {len(df_test):,} matches (skipped first 100 for ELO stability)")
    
    total_matches = 0
    higher_elo_wins = 0
    ties = 0
    
    p1_higher_and_wins = 0
    p2_higher_and_wins = 0
    p1_higher_total = 0
    p2_higher_total = 0
    
    for _, row in df_test.iterrows():
        p1_elo = row['elo_p1']
        p2_elo = row['elo_p2']
        target = row['target']
        
        if pd.isna(p1_elo) or pd.isna(p2_elo):
            continue
            
        total_matches += 1
        
        if p1_elo > p2_elo:
            p1_higher_total += 1
            if target == 0:
                higher_elo_wins += 1
                p1_higher_and_wins += 1
                
        elif p2_elo > p1_elo:
            p2_higher_total += 1
            if target == 1:
                higher_elo_wins += 1
                p2_higher_and_wins += 1
                
        else:
            ties += 1
    
    if total_matches > 0:
        overall_accuracy = (higher_elo_wins / total_matches) * 100
        p1_accuracy = (p1_higher_and_wins / p1_higher_total) * 100 if p1_higher_total > 0 else 0
        p2_accuracy = (p2_higher_and_wins / p2_higher_total) * 100 if p2_higher_total > 0 else 0
        tie_percentage = (ties / total_matches) * 100
        
        print(f"\nRESULTS:")
        print(f"   Total matches analyzed: {total_matches:,}")
        print(f"   ELO ties: {ties:,} ({tie_percentage:.1f}%)")
        print(f"\nHIGHER ELO PLAYER WIN RATE:")
        print(f"   Overall accuracy: {overall_accuracy:.1f}%")
        print(f"   ({higher_elo_wins:,} wins out of {total_matches:,} matches)")
        
        print(f"\nBREAKDOWN:")
        print(f"   Player 1 higher ELO: {p1_higher_total:,} times")
        print(f"   Player 1 won: {p1_higher_and_wins:,} times ({p1_accuracy:.1f}%)")
        print(f"   Player 2 higher ELO: {p2_higher_total:,} times") 
        print(f"   Player 2 won: {p2_higher_and_wins:,} times ({p2_accuracy:.1f}%)")
        
        print(f"\nINTERPRETATION:")
        if overall_accuracy > 60:
            print(f"   Excellent! ELO system shows strong predictive power ({overall_accuracy:.1f}%)")
        elif overall_accuracy > 55:
            print(f"   Good! ELO system has decent predictive power ({overall_accuracy:.1f}%)")
        elif overall_accuracy > 50:
            print(f"   Weak signal. ELO system has limited predictive power ({overall_accuracy:.1f}%)")
        else:
            print(f"   Poor. ELO system shows no predictive power ({overall_accuracy:.1f}%)")
            
        print(f"   Note: Random guessing would be ~50%. Your system is {overall_accuracy-50:.1f}% better than random.")
        
        return overall_accuracy
        
    else:
        print("No valid matches found for analysis")
        return 0


def test_elo_accuracy_by_year():
    """
    Test ELO accuracy broken down by year to see if it improves over time
    as the system learns more about players.
    """
    
    print(f"\n" + "=" * 70)
    print("ELO ACCURACY BY YEAR")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'with_features.csv'
    
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
    except FileNotFoundError:
        print("Could not load data for yearly analysis")
        return
    
    yearly_stats = []
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].iloc[10:].copy()
        
        if len(year_df) == 0:
            continue
            
        total = 0
        correct = 0
        
        for _, row in year_df.iterrows():
            p1_elo = row['elo_p1']
            p2_elo = row['elo_p2']
            target = row['target']
            
            if pd.isna(p1_elo) or pd.isna(p2_elo) or p1_elo == p2_elo:
                continue
                
            total += 1
            
            if (p1_elo > p2_elo and target == 0) or (p2_elo > p1_elo and target == 1):
                correct += 1
        
        if total > 0:
            accuracy = (correct / total) * 100
            yearly_stats.append({
                'year': year,
                'matches': total,
                'accuracy': accuracy,
                'correct': correct
            })
    
    if yearly_stats:
        print(f"{'Year':<6} {'Matches':<8} {'Accuracy':<10} {'Correct/Total'}")
        print("-" * 40)
        
        for stats in yearly_stats:
            print(f"{stats['year']:<6} {stats['matches']:<8} {stats['accuracy']:<10.1f}% {stats['correct']}/{stats['matches']}")
        
        early_years = [s['accuracy'] for s in yearly_stats[:5]]
        late_years = [s['accuracy'] for s in yearly_stats[-5:]]
        
        if early_years and late_years:
            early_avg = sum(early_years) / len(early_years)
            late_avg = sum(late_years) / len(late_years)
            
            print(f"\nTREND ANALYSIS:")
            print(f"   Early years average: {early_avg:.1f}%")
            print(f"   Recent years average: {late_avg:.1f}%")
            print(f"   Improvement: {late_avg - early_avg:+.1f}%")


if __name__ == "__main__":
    overall_accuracy = test_elo_accuracy()
    test_elo_accuracy_by_year()
    
    print(f"\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
