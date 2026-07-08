import os
from sklearn.model_selection import train_test_split
from utils.helper import load_config, create_dirs_if_not_exist
from data.data_loader import check_and_generate_mock_data, load_rul_df
from models.rul_predictor.model import TireRULPredictor

def train_rul_model():
    """
    Train the multi-output XGBoost regression RUL model.
    Saves the trained model as a pickle file.
    """
    config = load_config()
    create_dirs_if_not_exist(config)
    
    # Ensure datasets exist
    check_and_generate_mock_data(config)
    
    # Load DataFrame
    df = load_rul_df(config)
    print(f"Loaded RUL dataset with shape: {df.shape}")
    
    # Apply subsampling if enabled (CPU-friendly feature)
    subsample_size = config["data"].get("rul_subsample_size", 100000)
    if subsample_size > 0 and len(df) > subsample_size:
        print(f"Subsampling dataset to {subsample_size} rows for faster CPU training...")
        df = df.sample(n=subsample_size, random_state=config["general"]["seed"]).reset_index(drop=True)
        
    # Split into features (X) and targets (y)
    targets = ["remaining_useful_life(km)", "current_tread_depth(mm)"]
    X = df.drop(columns=targets)
    y = df[targets]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["general"]["seed"]
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Instantiate predictor
    n_estimators = config["models"]["rul_predictor"]["n_estimators"]
    max_depth = config["models"]["rul_predictor"]["max_depth"]
    learning_rate = config["models"]["rul_predictor"]["learning_rate"]
    
    predictor = TireRULPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        seed=config["general"]["seed"]
    )
    
    print("Training XGBoost MultiOutputRegressor model...")
    predictor.train(X_train, y_train)
    
    # Save the model
    save_path = config["models"]["rul_predictor"]["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictor.save(save_path)
    print("RUL training completed.")

if __name__ == "__main__":
    train_rul_model()
