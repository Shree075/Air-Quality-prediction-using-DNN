# Air Quality PM2.5 Forecasting Project

This Jupyter notebook implements an end-to-end machine learning pipeline for PM2.5 air pollution forecasting using environmental data. It covers data preprocessing, outlier removal via IQR, feature engineering, DNN modeling (basic and improved), and baselines like Linear Regression, Random Forest, XGBoost.[file:1]

## Features
- **Data Prep**: Forward-fill missing values, time features (hour/day/month/weekday), outlier clipping/removal on top features (PM10, PM1.0, etc.).
- **Models**: Improved DNN (R²=0.9470, MAE=2.04), outperforms baselines.
- **Evaluation**: Metrics on ~390k test samples; visualizations (boxplots, correlations, predictions).[file:1]

## Performance Comparison
| Model           | MAE  | RMSE | R²    |
|-----------------|------|------|-------|
| Basic DNN      | 522.28 | 936.66 | -10.51 |
| Improved DNN   | 2.04  | 6.65  | 0.9470|
| Linear Reg.    | 5.28 | 14.64 | 0.7431|
| Random Forest  | 1.91 | 6.86  | 0.9435|
| XGBoost        | 2.28 | 8.30  | 0.9174|[file:1]

## Setup & Run
1. Install: `pip install numpy pandas scikit-learn tensorflow matplotlib seaborn xgboost joblib`
2. Run `A1.ipynb` in Jupyter/Colab – auto-saves models (`improved_dnn_model.h5`), scalers (`target_scaler.pkl`), test data (`X_test_scaled.npy`, `y_test.npy`).[file:1]

## Results
- Top features: PM10 (0.89 corr), PM1.0 (0.77), etc.
- Loss curves, actual vs predicted plots included.[file:1]

**License**: MIT
