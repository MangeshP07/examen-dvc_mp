import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalize_data():
    # 1. Define paths for processed data
    input_dir = 'data/processed_data'
    
    # 2. Load the split datasets
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # 3. Handle non-numerical columns (like Date/Time)
    # We will only scale columns that are numbers (float or int)
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

    # 4. Initialize the StandardScaler
    scaler = StandardScaler()
    
    # 5. Fit on numeric training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # 6. Save the scaled data back to the processed folder
    pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns).to_csv(f"{input_dir}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns).to_csv(f"{input_dir}/X_test_scaled.csv", index=False)

    print("Success! Data normalized (excluding date columns) and saved.")

if __name__ == "__main__":
    normalize_data()