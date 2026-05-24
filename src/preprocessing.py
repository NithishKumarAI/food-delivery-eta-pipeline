"""
Preprocessing for food delivery ETA inference.

Mirrors the feature engineering in notebooks/01_data_exploration.ipynb
before CatBoost training (no one-hot encoding; CatBoost handles categories).
"""

from __future__ import annotations

import pandas as pd
from geopy.distance import geodesic

# Same 20 columns the saved CatBoost model was trained on (order matters for clarity).
MODEL_FEATURE_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "day_of_week",
    "is_weekend",
    "preparation_time",
    "order_hour",
    "distance_km",
    "meal_time",
]

# Categorical columns passed to CatBoost during training.
CATEGORICAL_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "day_of_week",
    "meal_time",
]

# Raw datetime columns removed after features are extracted (same as training).
DATETIME_COLUMNS_TO_DROP = [
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
]


def _calculate_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Restaurant-to-delivery distance in km (geodesic, same as training notebook)."""
    return geodesic((lat1, lon1), (lat2, lon2)).km


def _assign_meal_time(hour: int) -> str:
    """Map order hour to meal period (matches training logic)."""
    if 6 <= hour < 11:
        return "Breakfast"
    if 11 <= hour < 16:
        return "Lunch"
    if 16 <= hour < 22:
        return "Dinner"
    return "LateNight"


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw order rows, then drop datetime columns.

    Expected raw inputs include coordinates, categoricals, Order_Date, Time_Orderd,
    and either preparation_time or Time_Order_picked (to compute preparation_time).
    """
    df = data.copy()

    # --- distance_km (geodesic, same as training) ---
    df["distance_km"] = df.apply(
        lambda row: _calculate_distance_km(
            row["Restaurant_latitude"],
            row["Restaurant_longitude"],
            row["Delivery_location_latitude"],
            row["Delivery_location_longitude"],
        ),
        axis=1,
    )

    # --- Parse order time (training used %H:%M:%S; inference accepts flexible strings) ---
    time_ordered = pd.to_datetime(df["Time_Orderd"], errors="coerce")
    df["order_hour"] = time_ordered.dt.hour

    # --- Parse order date (training used %d-%m-%Y; also accept ISO dates like 2024-05-26) ---
    order_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    missing_dates = order_date.isna()
    if missing_dates.any():
        order_date.loc[missing_dates] = pd.to_datetime(
            df.loc[missing_dates, "Order_Date"], errors="coerce"
        )
    df["day_of_week"] = order_date.dt.day_name()

    # --- is_weekend (boolean, same as training: Saturday / Sunday) ---
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    # --- meal_time from order hour ---
    df["meal_time"] = df["order_hour"].apply(_assign_meal_time)

    # --- preparation_time: use existing column or derive from pick-up time ---
    if "preparation_time" not in df.columns:
        if "Time_Order_picked" not in df.columns:
            raise ValueError(
                "Provide either 'preparation_time' or 'Time_Order_picked' "
                "so preparation_time can be computed."
            )
        time_picked = pd.to_datetime(df["Time_Order_picked"], errors="coerce")
        df["preparation_time"] = (time_picked - time_ordered).dt.total_seconds() / 60
    elif df["preparation_time"].isna().any() and "Time_Order_picked" in df.columns:
        time_picked = pd.to_datetime(df["Time_Order_picked"], errors="coerce")
        missing = df["preparation_time"].isna()
        df.loc[missing, "preparation_time"] = (
            time_picked[missing] - time_ordered[missing]
        ).dt.total_seconds() / 60

    # Remove raw datetime columns (model never saw these)
    df = df.drop(columns=DATETIME_COLUMNS_TO_DROP, errors="ignore")

    return df


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with exactly the columns and order CatBoost expects.

    Categoricals are cast to string so inference matches training dtypes.
    """
    missing = [col for col in MODEL_FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = data[MODEL_FEATURE_COLUMNS].copy()

    for col in CATEGORICAL_COLUMNS:
        features[col] = features[col].astype(str)

    return features
