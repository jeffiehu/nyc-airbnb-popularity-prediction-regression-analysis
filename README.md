# NYC Airbnb Popularity Prediction: Regression Analysis

## Project Overview
This project builds a machine learning model to predict how popular an Airbnb listing will be based on its features. I used a 2019 New York City Airbnb dataset from Kaggle and my goal is to figure out what features actually drive guest bookings and engagement. For platforms like Airbnb, predicting a listing's success before it even goes live can help hosts build more appealing listings and attract guests faster.

---

## What I Did
* **Exploratory Data Analysis:** Analyzed at distributions and found that the target variable (`reviews_per_month`) is heavily right-skewed, and prices have extreme outliers.
* **Feature Engineering:** Created new features like `days_since_last_review` (to measure listing "freshness"), `availability_ratio`, and log-transformed prices to handle skewness.
* **Model Comparison:** Tested a baseline Dummy model, Ridge Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost using 5-fold cross-validation.
* **Optimization & Selection:** Used RFECV for feature selection and RandomizedSearchCV to tune hyperparameters for XGBoost.

---

## Tech Stack & Libraries
* **Language:** Python 3.12
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning & Modeling:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomizedSearchCV`, `DummyRegressor`, `Ridge`, `KNeighborsRegressor`, `RandomForestRegressor`), XGBoost (`XGBRegressor`)
* **Feature Selection:** RFECV

---

## Results Summary

| Model | Validation MAE | Final Test MAE | Key Takeaways |
| :--- | :---: | :---: | :--- |
| **Linear Baseline** | 0.9012 | N/A | Underfitted; too simple for complex NYC trends. |
| **KNN** | 0.7348 | N/A | Improved accuracy over baseline, but slow prediction times. |
| **Random Forest** | 0.5421 | N/A | High accuracy, but slower training time (~10s per fold). |
| **XGBoost (Optimized)** | **0.5372** | **0.5384** | **Winner:** Fast, lean, robust, and most accurate. |

* **Final Test Score:** **0.5384 MAE** (demonstrating high consistency with validation scores and proving the model avoided optimization bias).

---

## Key Takeaways
* **Freshness Matters:** `days_since_last_review` was by far the most important predictor, showing that active listings get more engagement.
* **Tree Models Won:** Tree-based ensembles (XGBoost) easily outperformed linear models because they handled non-linear relationships and interactions much better.
* **Location is Key:** Prime spots like the Theater District heavily drove review counts.
