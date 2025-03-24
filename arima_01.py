# @madakixo 14012025-11:52 
#updated 2402024-18:03
"""
ARIMA model for forecasting GDP (current LCU) with robust differencing and fallback
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

# Load data
df = pd.read_csv('1960_onwards1.csv')

# Convert 'Year' to datetime and set as index
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Select time series
series = df['GDP (current LCU)']

# Check for invalid values (zeros, negatives, NaNs)
print("Checking for invalid values in series:")
print("Zeros:", (series == 0).sum())
print("Negatives:", (series < 0).sum())
print("NaNs:", series.isna().sum())

# Handle NaNs if any
series = series.fillna(method='ffill')  # Forward fill NaNs

# Log-transform the series to stabilize variance
# Ensure no zeros/negatives before logging
if (series <= 0).any():
    print("Series contains zeros or negatives, adding a small constant before log-transform...")
    series = series + 1e-6  # Small constant to avoid log(0)
series_log = np.log(series)

# Check for stationarity
def test_stationarity(timeseries, label="series"):
    result = adfuller(timeseries, autolag='AIC')
    print(f'\nStationarity test for {label}:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result[1]  # Return p-value

# Test original series (log-transformed)
p_value = test_stationarity(series_log, "log-transformed series")

# Apply differencing only if necessary
d = 0
series_transformed = series_log.copy()
if p_value >= 0.05:
    print("\nSeries is non-stationary, applying first differencing...")
    series_transformed = series_log.diff().dropna()
    d = 1
    p_value = test_stationarity(series_transformed, "first differenced series")
    if p_value >= 0.05:
        print("\nStill non-stationary, applying second differencing...")
        series_transformed = series_transformed.diff().dropna()
        d = 2
        test_stationarity(series_transformed, "second differenced series")

# Grid search for ARIMA parameters
p = q = range(0, 3)  # Keep p and q range
d_range = [d]  # Use the determined differencing order
pdq = list(product(p, d_range, q))

best_aic = np.inf
best_param = None
best_model = None

for param in pdq:
    try:
        mod = ARIMA(series_log, order=param)
        results = mod.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_param = param
            best_model = results
        print('ARIMA{} - AIC:{}'.format(param, results.aic))
    except Exception as e:
        print(f"ARIMA{param} failed: {e}")
        continue

if best_model is None:
    print("\nAll ARIMA models failed to converge. Trying a simpler model: ARIMA(1,1,1)...")
    best_param = (1, 1, 1)
    best_model = ARIMA(series_log, order=best_param).fit()

print(f'\nBest ARIMA{best_param} - AIC:{best_aic}')

# Forecast 5 years ahead (1976–1980)
forecast_log = best_model.forecast(steps=5)

# Exponentiate to revert log-transformation
forecast = np.exp(forecast_log)

# Create a DataFrame for the forecast with proper dates
forecast_dates = pd.date_range(start='1976-01-01', end='1980-01-01', freq='Y')
forecast_df = pd.DataFrame({'GDP (current LCU)': forecast}, index=forecast_dates)

print("\nForecasted GDP (current LCU) for 1976–1980:")
print(forecast_df)

# Plot historical data and forecast
plt.figure(figsize=(10, 6))
plt.plot(series, label='Historical GDP (current LCU)', color='blue')
plt.plot(forecast_df, label='Forecasted GDP (current LCU)', color='red', linestyle='--')
plt.title('GDP (current LCU) Forecast (1960–1980)')
plt.xlabel('Year')
plt.ylabel('GDP (current LCU)')
plt.legend()
plt.grid()
plt.show()

# Optional: Forecast to 2034 (if that's your intent)
forecast_log_long = best_model.forecast(steps=59)  # 1976 to 2034 (59 years)
forecast_long = np.exp(forecast_log_long)
forecast_dates_long = pd.date_range(start='1976-01-01', end='2034-01-01', freq='Y')
forecast_df_long = pd.DataFrame({'GDP (current LCU)': forecast_long}, index=forecast_dates_long)

print("\nForecasted GDP (current LCU) for 2025–2034 (subset of long-term forecast):")
print(forecast_df_long.loc['2025-01-01':'2034-01-01'])
