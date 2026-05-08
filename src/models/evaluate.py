import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def evaluate_model():
    # 1. Load scaled test data and the trained model
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')
    
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 2. Predict on test data
    predictions = model.predict(X_test)

    # 3. Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # 4. Save metrics to JSON file for DVC
    os.makedirs('metrics', exist_ok=True)
    metrics_data = {"mse": mse, "r2_score": r2}
    
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # 5. Save predictions to a CSV file in data/ folder
    pred_df = pd.DataFrame(predictions, columns=['predicted_silica_concentrate'])
    pred_df.to_csv('data/predictions.csv', index=False)

    print(f"Evaluation complete! R2 Score: {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()