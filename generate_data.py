"""
Generate wine dataset from sklearn's built-in dataset
This is the Wine Recognition Dataset from UCI Machine Learning Repository
"""
from sklearn.datasets import load_wine
import pandas as pd
import os


def generate_wine_data():
    """Generate wine.csv from sklearn dataset"""

    print("Generating wine cultivar dataset...")

    # Load the dataset from sklearn
    data = load_wine()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cultivar'] = data.target  # 0, 1, 2 representing different cultivars

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/wine.csv', index=False)

    print(f"✓ Created data/wine.csv with {len(df)} wine samples")
    print(f"  - Cultivar 1: {(df['cultivar'] == 0).sum()} samples")
    print(f"  - Cultivar 2: {(df['cultivar'] == 1).sum()} samples")
    print(f"  - Cultivar 3: {(df['cultivar'] == 2).sum()} samples")
    print(f"  - Features: {len(data.feature_names)}")


if __name__ == "__main__":
    generate_wine_data()