"""
🌊 Prophet-Based Forecasting for Coastal Water Quality Parameters (BOD & DO)
Author: Harshit Gupta
Purpose:
    Forecasts Biochemical Oxygen Demand (BOD) and Dissolved Oxygen (DO)
    for each unique monitoring station using Facebook Prophet.

Outputs:
    1. forecasts_summary.csv  — all forecasted values (2025–2026)
    2. high_risk_forecasts.csv — stations showing worsening trends
    3. /plots/ — PDF plots of BOD and DO forecasts with confidence intervals
"""

# =========================================================
# 📦 Imports
# =========================================================
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set global style for consistency
plt.style.use("seaborn-v0_8-whitegrid")


# =========================================================
# ⚙️ Utility: Create Output Directories
# =========================================================
def create_output_dirs():
    base_dir = os.path.join(os.getcwd(), "outputs_forecast")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return base_dir, plot_dir


# =========================================================
# 🧹 Data Loading and Validation
# =========================================================
def load_data(input_csv):
    print(f"📂 Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = ['station_code', 'location_name', 'year', 'bod', 'dissolved_oxygen']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    df.dropna(subset=['bod', 'dissolved_oxygen', 'year'], inplace=True)
    df['year'] = df['year'].astype(int)

    print(f"✅ Data loaded successfully with {df.shape[0]} records and {df['station_code'].nunique()} stations.\n")
    return df


# =========================================================
# 🔮 Forecast Function
# =========================================================
def forecast_parameter(df, param, plot_dir, future_years=[2025, 2026]):
    """
    Forecasts a single parameter (e.g., BOD or DO) using Prophet for each station.
    Returns: DataFrame of forecasts for all stations.
    """
    results = []
    print(f"📈 Forecasting {param.upper()} for all stations...")

    for station, group in tqdm(df.groupby('station_code')):
        if len(group['year'].unique()) < 3:
            # Not enough data to train Prophet
            continue

        # Prepare data for Prophet
        data = group[['year', param]].copy()
        data = data.rename(columns={'year': 'ds', param: 'y'})
        data['ds'] = pd.to_datetime(data['ds'], format='%Y')

        model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        model.fit(data)

        # Create future dataframe
        future_dates = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
        forecast = model.predict(future_dates)

        # Extract relevant fields
        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_result['station_code'] = station
        forecast_result['parameter'] = param
        forecast_result['location_name'] = group['location_name'].iloc[0]
        results.append(forecast_result)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(data['ds'].dt.year, data['y'], marker='o', label='Historical', color='blue')
        plt.plot(forecast['ds'].dt.year, forecast['yhat'], label='Forecast', color='darkorange', linestyle='--')
        plt.fill_between(
            forecast['ds'].dt.year,
            forecast['yhat_lower'], forecast['yhat_upper'],
            alpha=0.3, color='orange', label='Confidence Interval'
        )
        plt.title(f"{param.upper()} Forecast - {group['location_name'].iloc[0]}")
        plt.xlabel("Year")
        plt.ylabel(param.upper() + " (mg/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{param}_forecast_{station}.pdf"))
        plt.close()

    print(f"✅ Forecasting for {param.upper()} completed.\n")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# =========================================================
# 🚨 Identify High-Risk Stations
# =========================================================
def identify_high_risk(forecasts_df):
    """
    Identifies stations where:
      - BOD shows an upward trend (increasing pollution)
      - DO shows a downward trend (worsening oxygen levels)
    """
    if forecasts_df.empty:
        return pd.DataFrame()

    print("⚠️ Identifying high-risk stations based on forecast trends...")
    risks = []

    for (station, param), group in forecasts_df.groupby(['station_code', 'parameter']):
        group = group.sort_values('ds')
        if len(group) < 2:
            continue

        start_val = group.iloc[0]['yhat']
        end_val = group.iloc[-1]['yhat']
        trend = end_val - start_val

        if param == 'bod' and trend > 0:
            risks.append({
                "station_code": station,
                "location_name": group['location_name'].iloc[0],
                "parameter": "BOD ↑ (Increasing Pollution)",
                "change": round(trend, 2)
            })
        elif param == 'dissolved_oxygen' and trend < 0:
            risks.append({
                "station_code": station,
                "location_name": group['location_name'].iloc[0],
                "parameter": "DO ↓ (Decreasing Oxygen)",
                "change": round(trend, 2)
            })

    risks_df = pd.DataFrame(risks)
    print(f"✅ Identified {len(risks_df)} high-risk stations.\n")
    return risks_df


# =========================================================
# 🚀 Main Forecasting Pipeline
# =========================================================
def run_forecast_pipeline(input_csv):
    base_dir, plot_dir = create_output_dirs()
    df = load_data(input_csv)

    forecast_bod = forecast_parameter(df, 'bod', plot_dir)
    forecast_do = forecast_parameter(df, 'dissolved_oxygen', plot_dir)

    # Combine all forecasts
    all_forecasts = pd.concat([forecast_bod, forecast_do], ignore_index=True)
    summary_path = os.path.join(base_dir, "forecasts_summary.csv")
    all_forecasts.to_csv(summary_path, index=False)

    # Identify and save high-risk stations
    high_risk_df = identify_high_risk(all_forecasts)
    high_risk_path = os.path.join(base_dir, "high_risk_forecasts.csv")
    high_risk_df.to_csv(high_risk_path, index=False)

    print("🎯 Forecasting Completed Successfully!")
    print(f"📄 All Forecasts: {summary_path}")
    print(f"⚠️ High-Risk Stations: {high_risk_path}")
    print(f"📊 Plots saved to: {plot_dir}")


# =========================================================
# 🏁 Entry Point
# =========================================================
if __name__ == "__main__":
    # 👇👇👇 PLACE YOUR INPUT FILE PATH HERE 👇👇👇
    input_csv_path = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\cleaned_water_quality_data.csv'
    # Example:
    # input_csv_path = r"D:\Research\Data\cleaned_water_quality_data.csv"

    run_forecast_pipeline(input_csv_path)