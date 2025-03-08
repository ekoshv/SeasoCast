# SeasoCast
Seasonal Forecasting Framework
# Seasonal MSTL ARIMA Sine GARCH Forecasting Framework

This repository contains a comprehensive forecasting pipeline for stock and crypto market data. It integrates multiple advanced techniques to capture the complex behavior of time series data, including:

- **Seasonal Period Detection using FFT:** Identifies dominant seasonal cycles in the data.
- **MSTL Decomposition:** Decomposes the time series into trend, seasonal, and residual components.
- **ARIMA Trend Forecasting:** Models and forecasts the trend component.
- **Sine-Cosine Seasonal Forecasting:** Fits sine and cosine functions on the last complete cycle of the seasonal components to generate forecasts.
- **GARCH Residual Modeling:** Models residual volatility to construct 95% confidence intervals.
- **Hyperparameter Optimization (Optuna):** Provides an optional framework to tune parameters like window size, refit steps, and number of seasonal components.

The entire pipeline is implemented in a single Python script, and the repository also includes an Excel file containing historical crypto market data.

---

## Repository Contents

- **`forecasting_script.py`**  
  The main Python script that implements the entire forecasting framework.
  
- **`crypto_data_1D_2017-01-01_to_2025-01-18_20250118_091237.xlsx`**  
  Excel file with historical crypto market data (daily frequency).

---

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or higher installed. The following Python libraries are required:

- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- arch
- optuna
- openpyxl (for reading Excel files)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scipy statsmodels arch optuna openpyxl
