# vegetation-resilience-across-North-America

# Overview

This project contains Python and Matlab scripts for time series analysis and machine learning model training, including 1-lag autocorrelation (AR1) analysis, STL (Seasonal-Trend decomposition using LOESS) decomposition, XGBoost model training with hyperparameter tuning, SHAP (SHapley Additive exPlanations) model interpretation, Theil–Sen slope estimator, and Mann–Kendall test.

# Project Structure
├── STL_decomposition.py # STL decomposition for time series  
├── 1-lag autocorrelation analysis.py # AR1 autoregressive analysis  
├── Sen_MK_AR1.m # Theil–Sen slope, and Mann–Kendall test
├── example_run.m # runing file for Sen_MK_AR1.m 
├── XGBoost+SHAP+TAC.py # XGBoost training with SHAP feature importance analysis  
├── XGBoost+SHAP+deltaTAC.py # XGBoost training with SHAP feature importance analysis  
├── input/ # Input data directory  
└── output/ # Output results directory  
