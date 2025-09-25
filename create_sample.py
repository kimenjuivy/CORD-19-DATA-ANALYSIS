import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    print("Loading full dataset...")
    df = pd.read_csv('raw_data/metadata_cleaned.csv')
    
    print(f"Original dataset: {len(df):,} papers")
    
    # Create stratified sample - 500 papers per year to keep it manageable
    sample_dfs = []
    
    for year in sorted(df['publication_year'].unique()):
        year_data = df[df['publication_year'] == year]
        sample_size = min(500, len(year_data))  # Max 500 per year
        if len(year_data) > 0:
            year_sample = year_data.sample(n=sample_size, random_state=42)
            sample_dfs.append(year_sample)
            print(f"Year {int(year)}: {len(year_data):,} → {sample_size:,} papers")
    
    # Combine samples
    sample_df = pd.concat(sample_dfs, ignore_index=True)
    
    # Save sample
    sample_df.to_csv('raw_data/metadata_sample.csv', index=False)
    
    print(f"\n✅ Created sample dataset: {len(sample_df):,} papers")
    
    # Check file size
    size_mb = os.path.getsize('raw_data/metadata_sample.csv') / (1024 * 1024)
    print(f"📁 Sample file size: {size_mb:.1f} MB")
    
    return sample_df

if __name__ == "__main__":
    create_sample_dataset()