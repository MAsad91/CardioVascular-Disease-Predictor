import os
import pandas as pd
import urllib.request

def download_heart_disease_data():
    """
    Downloads the heart disease dataset from the UCI Machine Learning Repository
    and saves it to the data directory.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL for the heart disease dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    
    # Download the dataset
    print("Downloading heart disease dataset...")
    urllib.request.urlretrieve(url, 'data/heart_disease_raw.csv')
    
    # Load and process the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    df = pd.read_csv('data/heart_disease_raw.csv', header=None, names=column_names)
    
    # Handle missing values (indicated by '?' in the dataset)
    df = df.replace('?', float('nan'))
    
    # Convert columns to appropriate types
    for col in df.columns:
        if col != 'target':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Process target variable (values > 0 indicate presence of heart disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Save processed dataset
    df.to_csv('data/heart_disease.csv', index=False)
    print(f"Dataset downloaded and processed. Total samples: {len(df)}")
    
    return df

if __name__ == "__main__":
    download_heart_disease_data() 