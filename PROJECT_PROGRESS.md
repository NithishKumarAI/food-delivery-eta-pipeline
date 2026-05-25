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

---

# Phase 5 - Advanced Model Training

Implemented Models:
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- LightGBM Regressor

Evaluation Summary:

| Model | R² Score |
|---|---|
| Linear Regression | 0.60 |
| Random Forest | 0.82 |
| XGBoost | 0.82 |
| LightGBM | 0.83 |
| CatBoost | 0.84 |

Final Production Model:
- CatBoost Regressor

Why CatBoost?
- Best overall performance
- Better categorical feature handling
- Smaller deployment size
- Faster inference
- More deployment-friendly than Random Forest

---

# Phase 6 - Inference Pipeline Engineering

Created reusable ML backend modules:

```text
src/
├── preprocessing.py
├── predict.py
└── main.py

---

# Phase 7 - Streamlit Frontend Development

Built an interactive frontend using Streamlit.

Features:
- User-friendly form inputs
- Real-time ETA prediction
- Integrated preprocessing + model inference
- Error handling for invalid inputs
- Lightweight UI for deployment demos

Frontend Flow:

```text
User Input → Streamlit Form → Preprocessing → CatBoost Prediction → ETA Output
```

---

# Phase 8 - Docker Containerization

Containerized the entire application using Docker.

Implemented:
- Python 3.11 slim image
- requirements.txt dependency management
- Streamlit container startup configuration
- Port exposure for Cloud Run compatibility

Key Learnings:
- Docker image optimization
- Dependency management inside containers
- Linux package compatibility issues
- Importance of reproducible environments

---

# Phase 9 - Google Cloud Deployment

Deployed the application to Google Cloud Run.

Deployment Workflow:

```text
Local Development
    ↓
Docker Build
    ↓
Artifact Registry Push
    ↓
Google Cloud Run Deployment
```

Completed:
- Google Artifact Registry setup
- Docker image push
- Cloud Run deployment
- HTTPS public endpoint configuration
- Debugged deployment/runtime issues

Challenges Faced:
- PortAudio dependency issues
- Container startup failures
- Streamlit port binding configuration
- Environment-specific dependency problems

Key Learnings:
- Cloud deployment differs significantly from local execution
- Production debugging requires infrastructure understanding
- Container logs are critical for troubleshooting
- Deployment engineering is a major part of ML systems

---

# Final Production Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
Preprocessing Pipeline
  ↓
CatBoost Model
  ↓
ETA Prediction Output
```

---

# Final Outcomes

Successfully built:
- End-to-end ML pipeline
- Production-ready inference workflow
- Cloud-deployed ML application
- Containerized deployment architecture

Technologies Used:
- Python
- Pandas
- Scikit-learn
- CatBoost
- Streamlit
- Docker
- Google Cloud Run

---

# Future Improvements

Potential Enhancements:
- MLflow experiment tracking
- CI/CD pipeline
- Kubernetes deployment
- Real-time monitoring
- Automated retraining pipeline
- Feature store integration
- API authentication
- Database integration

---

# Overall Learning Summary

This project helped develop practical understanding of:
- Data preprocessing
- Feature engineering
- Model evaluation
- Production inference pipelines
- Containerization
- Cloud deployment
- Debugging production systems
- End-to-end ML engineering workflows

Most important realization:
Building a model is only one part of production ML systems. Deployment, infrastructure, debugging, and reproducibility are equally important.