import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, OptimizeWarning
import pywt  # For wavelet filtering
import warnings

# Holt's linear method for the trend
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import periodogram

# GARCH model
from arch import arch_model

# For checking error structure
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# -------------------------------
# Utility Functions
# -------------------------------

def get_top_seasonal_periods(data, num_periods=3):
    """
    Compute the FFT of the centered data and find the top num_periods frequencies.
    The mean is subtracted before FFT to remove the DC component.
    
    Returns the corresponding seasonal periods (in days) as unique integers.
    """
    data_centered = data - np.mean(data)
    n = len(data_centered)
    fft_vals = np.fft.fft(data_centered)
    fft_freq = np.fft.fftfreq(n, d=1)  # assume daily data (d=1)
    mask = fft_freq > 0  # only positive frequencies
    freqs = fft_freq[mask]
    fft_vals = fft_vals[mask]
    magnitudes = np.abs(fft_vals)
    peaks, _ = find_peaks(magnitudes)
    sorted_peaks = peaks[np.argsort(magnitudes[peaks])][::-1]
    
    unique_periods = []
    for peak in sorted_peaks:
        f = freqs[peak]
        if f > 0:
            period = 1.0 / f
            # Only consider periods between 2 days and half the window length
            if period >= 2 and period <= n/2:
                p_int = int(round(period))
                if p_int not in unique_periods:
                    unique_periods.append(p_int)
                    if len(unique_periods) == num_periods:
                        break
    return unique_periods

def forecast_seasonal_sine(seasonal_series, T, steps):
    """
    Fit a sine-cosine function to the last full cycle (length T) of the seasonal component
    and forecast for the next 'steps' days.
    
    The function is defined as:
       f(t) = A*sin(2*pi*t/T) + B*cos(2*pi*t/T) + offset
    """
    # Get the last full cycle and ensure we work with 1D arrays.
    last_cycle = seasonal_series.iloc[-T:]
    t = np.arange(T).astype(float)
    y = last_cycle.values.ravel()
    # Initial guesses
    A0 = (np.max(y) - np.min(y)) / 2
    offset0 = np.mean(y)
    B0 = 0.0

    def sine_cosine_func(t, A, B, offset, T_val):
        return A * np.sin(2 * np.pi * t / T_val) + B * np.cos(2 * np.pi * t / T_val) + offset

    func = lambda t, A, B, offset: sine_cosine_func(t, A, B, offset, T)
    lb = [-2*np.abs(A0), -2*np.abs(A0), offset0 - 2*np.abs(A0)]
    ub = [2*np.abs(A0), 2*np.abs(A0), offset0 + 2*np.abs(A0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(func, t, y, p0=[A0, B0, offset0], bounds=(lb, ub), maxfev=10000)
        except Exception as e:
            print(f"*** {e}")
            popt = [A0, B0, offset0]
    t_forecast = np.arange(T, T + steps).astype(float)
    forecast_values = func(t_forecast, *popt)
    forecast_index = pd.date_range(start=seasonal_series.index[-1] + pd.Timedelta(days=1),
                                   periods=steps, freq='D')
    return pd.Series(forecast_values, index=forecast_index)

# -------------------------------
# Main Class
# -------------------------------
class SeasonalWaveletHoltSineGARCH:
    def __init__(self, data, window_size, refit_steps,
                 num_seasonal=3,
                 wavelet='db3', wavelet_levels=3, log_en=True):
        """
        Rolling forecasting framework that:
          - Uses multi-level wavelet filtering to decompose the data into a smooth trend (approximation) and residual components.
          - Applies FFT on the residuals to detect dominant seasonal periods.
          - Forecasts the trend component using Holt’s linear method.
          - Forecasts the residual (seasonal) component using a sine-cosine function fitted on its last full cycle.
          - Combines the trend and seasonal forecasts for the overall deterministic forecast.
          - Uses a GARCH model on the wavelet residuals to build 95% confidence intervals.
        
        Parameters:
          data (pd.DataFrame): Time series data with a 'close' column and datetime index.
          window_size (int): Number of days used for training in each rolling window.
          refit_steps (int): Number of days ahead to forecast (and the refit interval).
          num_seasonal (int): Number of seasonal peak frequencies to detect.
          wavelet (str): Wavelet type to be used for filtering.
          wavelet_levels (int): Number of decomposition levels for a smoother trend.
          log_en (bool): If True, print logging messages.
        """
        self.df = data.copy()
        self.window_size = window_size
        self.refit_steps = refit_steps
        self.num_seasonal = num_seasonal
        self.log_en = log_en
        self.forecast_results = []
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        
    def wavelet_smooth(self, data):
        """
        Apply multi-level discrete wavelet transform to smooth the data.
        This function decomposes the data to a specified number of levels and 
        reconstructs the approximation by discarding the detail coefficients.
        
        Parameters:
            data (np.array): 1D array of the original data.
        
        Returns:
            trend (np.array): The smooth (approximation) component.
            residual (np.array): The difference between the original data and the trend.
        """
        orig_len = len(data)
        # Decompose the data using multi-level wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=self.wavelet_levels)
        # Zero out all detail coefficients for reconstruction
        approx_coeffs = [coeffs[0]] + [np.zeros_like(detail) for detail in coeffs[1:]]
        trend = pywt.waverec(approx_coeffs, self.wavelet)
        trend = trend[:orig_len]
        residual = data - trend
        return trend, residual
    
    def run_forecast(self, plot=True):
        n_iterations = int(np.ceil((len(self.df) - self.window_size) / self.refit_steps))
        cmap = plt.get_cmap('viridis', n_iterations)

        if plot:
            plt.figure(figsize=(12, 8), dpi=600)
            plt.plot(self.df.index, self.df['close'],
                     label="Historical Close", color="black", linewidth=0.5)

        for i in range(n_iterations):
            if self.log_en:
                print(f"--- Rolling window step {i+1} ---")
            train_end = self.window_size + i * self.refit_steps
            if train_end >= len(self.df):
                break

            train_data = self.df['close'].iloc[train_end - self.window_size:train_end]
            steps = self.refit_steps if (train_end + self.refit_steps) <= len(self.df) else len(self.df) - train_end

            # Step 1: Apply multi-level wavelet smoothing to obtain trend and residuals
            if self.log_en:
                print(train_data.size)
            trend_approx, residual = self.wavelet_smooth(train_data.values)
            trend_series = pd.Series(trend_approx, index=train_data.index)
            residual_series = pd.Series(residual, index=train_data.index)

            # Step 2: Detect seasonal periods from the residuals via FFT
            seasonal_periods = get_top_seasonal_periods(residual, num_periods=self.num_seasonal)
            if self.log_en:
                print("Detected seasonal periods (days):", seasonal_periods)
            if len(seasonal_periods) < self.num_seasonal:
                if self.log_en:
                    print("Not enough seasonal periods detected; using available periods:", seasonal_periods)

            # Step 3: Forecast Trend using Holt's linear method on the trend (approximation) component
            try:
                # ------------------------------------------------
                # 1) DETREND USING LOCAL REGRESSION (LOESS / LOWESS)
                # ------------------------------------------------
                # Create a numeric index
                x = np.arange(len(trend_series))

                # frac controls the smoothing window (as a fraction of the total data)
                # Smaller frac -> less smoothing, larger frac -> more smoothing
                loess_result = lowess(trend_series, x, frac=0.3)

                # lowess_result is a 2D array: first column is x, second column is the smoothed values
                trend_est = pd.Series(index=trend_series.index, data=loess_result[:, 1])

                # Detrend the series
                ts_detrended = trend_series - trend_est
                
                # ------------------------------------------------
                # 2) ESTIMATE SEASONAL PERIOD (PERIODOGRAM ON DETRENDED DATA)
                # ------------------------------------------------
                fs = 1  # sampling frequency
                freqs, power = periodogram(ts_detrended, fs=fs)
                # Ignore the zero frequency and find the frequency with the highest power
                idx = np.argsort(power[1:])[-1] + 1
                peak_freq = freqs[idx]
                estimated_period = int(round(1 / peak_freq))
                print("Estimated seasonal period of Trend:", estimated_period)

                # ------------------------------------------------
                # 3) FIT HOLT-WINTERS MODEL WITH ESTIMATED SEASONAL PERIOD
                # ------------------------------------------------
                
                trend_model = ExponentialSmoothing(trend_series,
                                                   trend='add', 
                                                   damped_trend=True,
                                                   seasonal='add',
                                                   seasonal_periods=estimated_period,
                                                   
                                                   )
                trend_fit = trend_model.fit(optimized=True)
                trend_forecast = trend_fit.forecast(steps)
            except Exception as e:
                if self.log_en:
                    print(f"Trend Holt's method failed at step {i+1}: {e}")
                last_val = trend_series.iloc[-1]
                trend_forecast = pd.Series(
                    np.full(steps, last_val),
                    index=pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1),
                                        periods=steps, freq='D')
                )

            # Step 4: Forecast Seasonal Components via sine-cosine functions on the residuals
            seasonal_forecast = None
            for T in seasonal_periods:
                sine_forecast = forecast_seasonal_sine(residual_series, T, steps)
                if seasonal_forecast is None:
                    seasonal_forecast = sine_forecast
                else:
                    seasonal_forecast = seasonal_forecast + sine_forecast

            # Step 5: Combine Trend and Seasonal Forecasts
            overall_forecast = trend_forecast + (seasonal_forecast if seasonal_forecast is not None else 0)

            # Step 6: Forecast Residual Volatility via GARCH using the wavelet residuals
            try:
                garch_model = arch_model(residual_series, vol='Garch', p=2, q=2, o=1,
                                         power=2.0, dist='skewt', rescale=False)
                garch_fit = garch_model.fit(disp='off')
                garch_forecast = garch_fit.forecast(horizon=steps)
                forecast_variance = garch_forecast.variance.iloc[-1]
                forecast_vol = np.sqrt(forecast_variance)
                lower_forecast = overall_forecast - 1.96 * forecast_vol.values
                upper_forecast = overall_forecast + 1.96 * forecast_vol.values
            except Exception as e:
                if self.log_en:
                    print(f"GARCH modeling failed at step {i+1}: {e}")
                lower_forecast = overall_forecast * 0.95
                upper_forecast = overall_forecast * 1.05

            forecast_dates = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1),
                                           periods=steps, freq='D')
            
            if plot:
                color = cmap(i)
                # Plot the trend
                plt.plot(trend_series.index, trend_series,
                         color='red', linewidth=0.5, alpha=0.2)
                # Plot the forecast
                plt.plot(forecast_dates, overall_forecast, color=color, linewidth=0.5,
                         label="Forecast" if i == 0 else None)
                # Vertical line for forecast start
                plt.axvline(x=train_data.index[-1] + pd.Timedelta(days=1),
                            color="gray", linestyle="--",
                            linewidth=0.3, alpha=0.25
                            )
                ## Confidence interval
                # plt.fill_between(forecast_dates, lower_forecast, upper_forecast,
                #                  color=color, alpha=0.20)

            temp_df = pd.DataFrame({
                "forecast": overall_forecast,
                "lower": lower_forecast,
                "upper": upper_forecast
            }, index=forecast_dates)
            self.forecast_results.append(temp_df)

        if plot:
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.title(f"Rolling Forecast: Wavelet, Holt's Trend, Sine Seasonal, and GARCH\n"
                      f"(Window={self.window_size}, Refit={self.refit_steps}, "
                      f"Seasons={self.num_seasonal}, Levels={self.wavelet_levels})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def get_rmse(self):
        if self.forecast_results:
            forecast_df = pd.concat(self.forecast_results).sort_index()
            merged_df = forecast_df.join(self.df["close"], how="inner")
            merged_df.rename(columns={"close": "actual"}, inplace=True)
            merged_df["residual"] = merged_df["forecast"] - merged_df["actual"]
            rmse = np.sqrt(np.mean(merged_df["residual"] ** 2))
            return rmse
        else:
            return float("inf")

    def evaluate_performance(self):
        if self.forecast_results:
            forecast_df = pd.concat(self.forecast_results).sort_index()
            merged_df = forecast_df.join(self.df["close"], how="inner")
            merged_df.rename(columns={"close": "actual"}, inplace=True)
            merged_df["residual"] = merged_df["forecast"] - merged_df["actual"]
            mae = np.mean(np.abs(merged_df["residual"]))
            mse = np.mean(merged_df["residual"] ** 2)
            rmse = np.sqrt(mse)
            if self.log_en:
                print("Forecast Performance:")
                print(f"MAE: {mae:.2f}")
                print(f"MSE: {mse:.2f}")
                print(f"RMSE: {rmse:.2f}")
        else:
            if self.log_en:
                print("No forecast results to evaluate.")
                
    def plot_normalized_error_timeseries(self):
        if not self.forecast_results:
            print("No forecast results to compute errors.")
            return

        forecast_df = pd.concat(self.forecast_results).sort_index()        
        merged_df = forecast_df.join(self.df["close"], how="inner")
        merged_df.rename(columns={"close": "actual"}, inplace=True)        
        merged_df["error_normalized"] = (merged_df["forecast"] - merged_df["actual"]) / merged_df["actual"]       
        self.error_timeseries = merged_df["error_normalized"]

        plt.figure(figsize=(12, 6), dpi=600)
        plt.plot(merged_df.index, merged_df["error_normalized"],
                 marker='o', markersize=2.0,
                 linestyle='-', label="Normalized Forecast Error")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
        plt.xlabel("Date")
        plt.ylabel("Error ((Forecast - Actual)/Actual)")
        plt.title("Concatenated Forecast Errors (by Actual Values)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_es_error_timeseries(self):
        if not self.forecast_results:
            print("No forecast results to compute errors.")
            return

        forecast_df = pd.concat(self.forecast_results).sort_index()       
        merged_df = forecast_df.join(self.df["close"], how="inner")
        merged_df.rename(columns={"close": "actual"}, inplace=True)
        merged_df["es_error_normalized"] = np.where(
            merged_df["forecast"] != 0,
            (merged_df["forecast"] - merged_df["actual"]) / merged_df["forecast"],
            np.nan
        )
        self.es_error_timeseries = merged_df["es_error_normalized"]

        plt.figure(figsize=(12, 6), dpi=600)
        plt.plot(merged_df.index, merged_df["es_error_normalized"],
                 marker='o', markersize=2.0,
                 linestyle='-', label="Error/Predicted")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
        plt.xlabel("Date")
        plt.ylabel("Error ((Forecast - Actual)/Forecast)")
        plt.title("Concatenated Forecast Errors (by Predicted Values)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_error_series(self):
        """
        Returns a pandas Series of the raw forecast errors:
           error = (forecast - actual).
        """
        if not self.forecast_results:
            return pd.Series(dtype=float)
        forecast_df = pd.concat(self.forecast_results).sort_index()
        merged_df = forecast_df.join(self.df["close"], how="inner")
        merged_df.rename(columns={"close": "actual"}, inplace=True)
        merged_df["error"] = merged_df["forecast"] - merged_df["actual"]
        return merged_df["error"]

# -------------------------------
# Function to Analyze Errors
# -------------------------------
def analyze_forecast_errors(error_series):
    """
    Analyze forecast errors to determine if an ARIMA-based error correction model 
    might be beneficial. Checks include:
      - Histogram & QQ-Plot
      - ACF & PACF plots
      - Ljung-Box test for autocorrelation
    """

    # 1. Histogram of errors
    plt.figure(figsize=(12, 5))
    plt.hist(error_series, bins=30, edgecolor='k')
    plt.title("Histogram of Forecast Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 2. QQ-Plot of errors
    plt.figure(figsize=(6, 5))
    sm.qqplot(error_series, line='45', fit=True)
    plt.title("QQ Plot of Forecast Errors")
    plt.show()

    # 3. ACF & PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(error_series, ax=axes[0], lags=50)
    axes[0].set_title("Autocorrelation (ACF) of Forecast Errors")
    plot_pacf(error_series, ax=axes[1], lags=50, method='ywm')
    axes[1].set_title("Partial Autocorrelation (PACF) of Forecast Errors")
    plt.tight_layout()
    plt.show()

    # 4. Ljung-Box test for autocorrelation
    #    If p-values are low (< 0.05), it suggests autocorrelation remains in the errors.
    lb_results = acorr_ljungbox(error_series, lags=[5, 10, 15, 20], return_df=True)
    print("Ljung-Box Test Results (for lags 5, 10, 15, 20):")
    print(lb_results)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Data Loading
    excel_file = "crypto_data_1D_2017-01-01_to_2025-01-18_20250118_091237.xlsx"
    sheet_name = "ADA"
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.columns = df.columns.str.lower()
    if "datetime" not in df.columns:
        raise ValueError("Error: 'datetime' column not found in the Excel file.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.asfreq("D")  # Ensure daily frequency

    # Create and run the model
    best_model = SeasonalWaveletHoltSineGARCH(
        data=df,
        window_size=120,
        refit_steps=14,
        num_seasonal=7,
        wavelet_levels=4,
        log_en=True
    )

    # Run the rolling forecast
    best_model.run_forecast(plot=True)

    # Evaluate performance
    best_model.evaluate_performance()

    # Plot error timeseries
    best_model.plot_normalized_error_timeseries()
    best_model.plot_es_error_timeseries()

    # ----------------------------------------------------
    #  Get the raw error series and analyze it for patterns
    # ----------------------------------------------------
    error_series = best_model.get_error_series()
    analyze_forecast_errors(error_series)
