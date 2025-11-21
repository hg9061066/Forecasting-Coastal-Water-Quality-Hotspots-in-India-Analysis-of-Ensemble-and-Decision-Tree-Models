# Coastal Water Quality Hotspot Forecasting Framework 🌊

A Machine Learning Framework for Forecasting Coastal Water Quality Hotspots in India: A Comparative Analysis of Ensemble and Decision Tree Models

📌 Overview

This repository contains the code and methodology for a machine learning framework designed to identify and forecast coastal water quality "hotspots" in India. The project supports UN Sustainable Development Goal 14 (Life Below Water) by transitioning environmental monitoring from reactive analysis to proactive, predictive governance.

The framework consists of two core components:

Classification Engine: A Random Forest-based model to identify current pollution hotspots with high accuracy (F1-Score: 0.992).

Forecasting Engine: A Facebook Prophet time-series model to predict future degradation trends for 2025-2026.

📂 Repository Structure

├── data/
│   ├── cleaned_water_quality_data.csv    # Aggregated annual dataset (2020-2023)
│   └── output/                           # Generated forecasts and metrics
├── src/
│   ├── master_ml_framework_final.py      # Classification (RF, XGBoost, DT)
│   └── forecast_prophet_final.py         # Time-series forecasting (Prophet)
├── notebooks/                            # Exploratory Data Analysis (EDA)
├── requirements.txt                      # Dependencies
└── README.md
