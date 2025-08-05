# Housing Price Prediction 📊🏠

This project uses multiple regression models to predict housing prices. Evaluation metrics include MAE, MSE, and R² across models like Linear Regression, Random Forest, XGBoost, and more.

## Files Included
- `model.py`: All model training and evaluation code
- `model_evaluation_results.csv`: Comparison of model performance

## How to Run
1. Clone the repo
2. Install required libraries using: `pip install -r requirements.txt`
3. Run `model.py` to train models and generate results

## Dataset
Sample housing dataset used.


---

## 📊 Dataset Overview

- **Features**: Avg. Area Income, Avg. House Age, Avg. Number of Rooms, Population, etc.
- **Target**: `Price` – housing price in USD

---

## ⚙️ Models Trained

| Model               | Type                |
|---------------------|---------------------|
| LinearRegression     | Linear              |
| Ridge, Lasso, ElasticNet | Regularized Linear |
| HuberRegressor       | Robust              |
| Polynomial (Degree=4)| Non-linear          |
| SGDRegressor         | Online Learning     |
| MLPRegressor         | Neural Network      |
| RandomForest         | Ensemble (Trees)    |
| SVR                  | Support Vector      |
| LGBM, XGBoost        | Gradient Boosting   |
| KNN                  | Distance-Based      |

---

## 📈 Evaluation Metrics

Each model is evaluated on:

| Metric | Meaning |
|--------|---------|
| **MAE** (Mean Absolute Error) | Avg. error in predictions. Lower is better. |
| **MSE** (Mean Squared Error) | Penalizes larger errors more than MAE. |
| **R² Score** | Measures goodness-of-fit. <br>**1 = perfect**, **0 = baseline**, **< 0 = worse than baseline**. |

---

## 📉 Results: Interpreting Metrics

| Model              | MAE         | MSE         | R² Score     | Interpretation |
|-------------------|-------------|-------------|--------------|----------------|
| LinearRegression   | 8.10e+04    | 1.01e+10    | 0.919        | ✅ Very good baseline |
| RidgeRegression    | ~8.1e+04    | ~1.01e+10   | ~0.919       | ✅ Similar to Linear |
| PolynomialRegression | Extremely high MAE & MSE | R² < 0 | ❌ Severely overfit |
| SGDRegressor       | MAE > 1e+18 | MSE > 1e+36 | R² = -1e+25  | ❌ Completely diverged |
| MLPRegressor       | Good MAE/R² | Around 0.93 | ✅ Performs well |
| RandomForest       | ~4.0e+04    | Lower MSE   | 0.965+       | ✅ Best performing |
| XGBoost / LGBM     | Similar to RF| ~0.96–0.97 | ✅ Robust models |
| SVR / KNN          | Higher error | R² ~0.7     | ⚠️ Weak fit |
| Huber              | Similar to Linear | ~0.91 | ✅ Robust baseline alternative |

> 🔍 **R² < 0** (e.g. in Polynomial, SGD) means the model performed worse than a simple average prediction. Likely due to overfitting, bad convergence, or poor hyperparameters.

---

## 🧠 Key Insights

- **RandomForest, XGBoost, and LGBM** had the best overall performance with high R² and low errors.
- **Polynomial Regression** and **SGDRegressor** overfit or diverged. Not suitable without tuning.
- **Regularized models (Ridge, Lasso)** perform similarly to baseline linear regression.
- **MLP (Neural Net)** is decent, but slower to train.

---
