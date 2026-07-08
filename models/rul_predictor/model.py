import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class TireRULPredictor:
    """
    XGBoost-based tabular regression model to predict remaining useful life (km)
    and current tread depth (mm) from vehicle, tire, and operational parameters.
    """
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        
        self.categorical_cols = [
            "vehicle_model", "fuel_type", "transmission_type", "country",
            "axle_type(driven/dead)", "tyre_brand", "tyre_size", 
            "tread_material", "tread_pattern", "retreaded", 
            "road_condition", "weather_condition"
        ]
        
        self.numeric_cols = [
            "maximum_power(hp)", "maximum_torque(N/m)", "maximum_speed(km/h)",
            "vehicle_acceleration(0-100 km/h in seconds)", "vehicle_mileage(mpg)",
            "vehicle_sprung_mass(kg)", "steering_radius(m)", "tyre_camber_angle(degree)",
            "standard_tread_depth(mm)", "expected_tyre_life(km)", "kilometers_driven(km)"
        ]
        
        self.pipeline = None

    def _build_pipeline(self):
        """
        Build scikit-learn preprocessing + multi-output XGBoost regression pipeline.
        """
        # Preprocessor for numeric and categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols)
            ]
        )
        
        # XGBoost regressor wrapped in MultiOutputRegressor
        xgb = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=-1
        )
        
        multi_regressor = MultiOutputRegressor(xgb)
        
        # Assemble pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", multi_regressor)
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Train the model pipeline.
        X: DataFrame of features
        y: DataFrame with columns ['remaining_useful_life(km)', 'current_tread_depth(mm)']
        """
        self._build_pipeline()
        
        # Keep only required features
        X_filtered = X[self.numeric_cols + self.categorical_cols]
        
        self.pipeline.fit(X_filtered, y)
        print("Tabular RUL Predictor pipeline trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict targets. Returns shape (n_samples, 2).
        First column: remaining_useful_life(km)
        Second column: current_tread_depth(mm)
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline has not been trained or loaded yet.")
            
        X_filtered = X[self.numeric_cols + self.categorical_cols]
        return self.pipeline.predict(X_filtered)

    def save(self, filepath: str):
        """
        Pickle the trained model pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save untrained model.")
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load a pickled model pipeline.
        """
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"Model loaded from {filepath}")
