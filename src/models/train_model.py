import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_final_model():
    # 1. Load scaled training data
    # English comment: Loading the preprocessed training set
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')

    # 2. Initialize the model with best parameters from Step 3
    # English comment: Using parameters identified by Grid Search
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )

    # 3. Fit the model
    print("Training the final model...")
    model.fit(X_train, y_train.values.ravel())

    # 4. Save the trained model to models/ directory
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Success! Final model saved as models/model.pkl")

if __name__ == "__main__":
    train_final_model()