import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

def run_grid_search():
    # 1. Load the scaled training data
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')

    # 2. Define the model (Random Forest)
    rf = RandomForestRegressor(random_state=42)

    # 3. Define the parameters to test
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 4. Set up GridSearchCV
    # We use cv=3 (3-fold cross validation) to save time
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

    # 5. Execute the search
    grid_search.fit(X_train, y_train.values.ravel())

    # 6. Create models directory and save the best estimator/params
    os.makedirs('models', exist_ok=True)
    
    # Save the best parameters/model as a .pkl file
    with open('models/best_params.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    print(f"Success! Best parameters found: {grid_search.best_params_}")

if __name__ == "__main__":
    run_grid_search()