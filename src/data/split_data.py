import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    # Load the raw dataset
    raw_data_path = 'data/raw_data/raw.csv'
    df = pd.read_csv(raw_data_path)

    # Separate features (X) and target (y)
    # The last column is 'silica_concentrate'
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # Only the last column

    # Split the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the output directory if it doesn't exist
    output_dir = 'data/processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets into the processed folder
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Success! 4 files saved in {output_dir}")

if __name__ == "__main__":
    split_data()