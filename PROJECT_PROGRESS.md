# Food Delivery ETA Prediction - Project Progress

## Project Goal
Build an end-to-end ML pipeline to predict food delivery time using real-world delivery data.

---

# Phase 1 - Project Setup

Completed:
- GitHub repository setup
- PyCharm environment setup
- Virtual environment creation
- Dataset organization

Project Structure:
```text
data/
notebooks/
src/
models/
```

---

# Phase 2 - Data Exploration & Cleaning

Notebook:
```text
notebooks/01_data_exploration.ipynb
```

Completed:
- Loaded raw JSON/TXT dataset
- Converted dataset into pandas DataFrame
- Explored dataset structure
- Identified missing values
- Cleaned categorical columns
- Removed unnecessary text patterns
- Converted target variable to integer
- Handled missing values
- Saved cleaned dataset as CSV

Important preprocessing steps:
- Removed fake "NaN" string values
- Handled missing values using median/mode
- Dropped rows with missing `Time_Orderd`
- Removed unnecessary ID columns

---

# Phase 3 - Feature Engineering

Created new features:
- `day_of_week`
- `is_weekend`
- `preparation_time`
- `order_hour`

Converted:
- date columns → datetime
- time columns → datetime

Removed raw datetime columns after extracting useful features.

---

# Phase 4 - Machine Learning Baseline

Train-test split:
- 80% training
- 20% testing

Encoding:
- Applied One-Hot Encoding using `pd.get_dummies()`

Baseline model:
- Linear Regression

Evaluation Results:

| Metric | Value |
|---|---|
| MAE | 4.69 |
| MSE | 34.79 |
| RMSE | 5.89 |
| R² Score | 0.60 |

Observations:
- Baseline model performs reasonably well
- Model captures meaningful delivery patterns
- Further improvement possible using advanced models

---

# Next Steps

Planned:
- Random Forest Regressor
- XGBoost Regressor
- Hyperparameter tuning
- MLflow experiment tracking
- Model serialization
- FastAPI deployment
- Docker containerization
- Google Cloud deployment

---

# Key Learnings So Far

- Real-world datasets are messy
- Preprocessing is critical in ML
- Feature engineering significantly impacts performance
- Proper train-test splitting prevents data leakage
- Baseline models are important for benchmarking

## Model Comparison

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | 4.69 | 3.74 |
| RMSE | 5.89 | 4.77 |
| R² Score | 0.60 | 0.74 |

Observations:
- Random Forest significantly outperformed Linear Regression
- Nonlinear relationships exist in delivery prediction data
- Tree-based models are better suited for this problem
- Random Forest currently becomes the best baseline model