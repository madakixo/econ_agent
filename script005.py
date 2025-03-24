 # Import necessary libraries for data manipulation, visualization, and time series modeling
import pandas as pd  # For data handling and DataFrame operations
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting historical and forecasted data
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model for time series forecasting
from statsmodels.tsa.stattools import adfuller  # Augmented Dickey-Fuller test for stationarity
from itertools import product  # To generate combinations of ARIMA parameters (p, d, q)

# Load the dataset from a CSV file
# The file '1960_onwards1.csv' is expected to contain economic data starting from 1960
df = pd.read_csv('1960_onwards1.csv')

# Define column names for date and target variable for clarity and reusability
date_col = 'Year'  # Column containing the time index (years)
target_col = 'GDP growth (annual %)'  # Column to forecast (GDP growth rate)

# Preprocess the data
# Convert the 'Year' column to datetime format for proper time series handling
df[date_col] = pd.to_datetime(df[date_col], format='%Y')  # '%Y' indicates year-only format (e.g., 1960)
# Set 'Year' as the index of the DataFrame to enable time-based operations
df.set_index(date_col, inplace=True)

# Extract the target time series (GDP growth) for analysis
series = df[target_col]

# Define a function to test stationarity using the Augmented Dickey-Fuller (ADF) test
# Stationarity means the series has constant mean, variance, and no trend over time
def test_stationarity(timeseries, label="series"):
    """
    Test if a time series is stationary using the ADF test.
    
    Args:
        timeseries: pandas Series, the time series to test
        label: str, a label for printing results (e.g., 'original series')
    
    Returns:
        float: p-value from the ADF test
    """
    # Perform ADF test; autolag='AIC' selects lag length based on Akaike Information Criterion
    result = adfuller(timeseries, autolag='AIC')
    print(f'\nStationarity test for {label}:')
    print('ADF Statistic: %f' % result[0])  # Test statistic; more negative means more stationary
    print('p-value: %f' % result[1])  # p-value; < 0.05 suggests stationarity (reject null hypothesis)
    print('Critical Values:')  # Thresholds for rejecting the null hypothesis at different confidence levels
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))  # e.g., 1%, 5%, 10% critical values
    return result[1]  # Return p-value for decision-making

# Test stationarity of the original series
# If non-stationary (p-value >= 0.05), differencing will be applied
p_value = test_stationarity(series, "original series")

# Initialize differencing order (d) and create a copy of the series
d = 0  # Start with no differencing
series_transformed = series.copy()  # Copy to preserve original data

# Check if the series is non-stationary and apply differencing if needed
if p_value >= 0.05:  # p-value >= 0.05 fails to reject null hypothesis (non-stationary)
    print("\nSeries is non-stationary, applying first differencing...")
    # First differencing: subtract each value from the previous one to remove trend
    series_transformed = series.diff().dropna()  # dropna() removes NaN from the first row
    d = 1  # Update differencing order
    p_value = test_stationarity(series_transformed, "first differenced series")
    if p_value >= 0.05:  # Check if still non-stationary after first differencing
        print("\nStill non-stationary, applying second differencing...")
        # Second differencing: difference the already differenced series
        series_transformed = series_transformed.diff().dropna()
        d = 2  # Update differencing order
        test_stationarity(series_transformed, "second differenced series")

# Perform grid search to find the best ARIMA parameters (p, d, q)
# p: number of autoregressive terms, d: differencing order, q: number of moving average terms
p = q = range(0, 3)  # Test p and q values from 0 to 2 (small range for simplicity)
d_range = [d]  # Use the determined differencing order from stationarity tests
pdq = list(product(p, d_range, q))  # Generate all combinations of (p, d, q)

# Initialize variables to track the best model based on AIC (Akaike Information Criterion)
best_aic = np.inf  # Start with infinity; lower AIC indicates better model fit
best_param = None  # To store the best (p, d, q) combination

# Iterate through all parameter combinations to find the best ARIMA model
for param in pdq:
    try:
        # Fit ARIMA model with the current (p, d, q) combination
        mod = ARIMA(series, order=param)  # Use original series, as differencing is handled by 'd'
        results = mod.fit()  # Fit the model to the data
        # Compare AIC to find the best model
        if results.aic < best_aic:
            best_aic = results.aic
            best_param = param
        print('ARIMA{} - AIC:{}'.format(param, results.aic))  # Print AIC for each model
    except Exception as e:
        # Handle cases where the model fails to converge or encounters errors
        print(f"ARIMA{param} failed: {e}")
        continue  # Skip to the next combination

# Output the best model's parameters and AIC
print(f'\nBest ARIMA{best_param} - AIC:{best_aic}')

# Fit the best ARIMA model using the identified parameters
best_model = ARIMA(series, order=best_param).fit()

# Generate a forecast for the next 30 years
# steps=30 means forecasting 30 periods (years) ahead
forecast = best_model.forecast(steps=30)
print("\nForecast for next 30 years (GDP growth %):")
print(forecast)  # Display the forecasted values

# Visualize the historical data and forecast
plt.figure(figsize=(10, 6))  # Set figure size for readability
# Plot historical GDP growth data
plt.plot(series, label='Historical GDP Growth (%)', color='blue')
# Plot forecasted GDP growth; index extends from the last historical date
plt.plot(forecast, label='Forecasted GDP Growth (%)', color='red', linestyle='--')
plt.title('GDP Growth Forecast (1960–2054)')  # Title of the plot
plt.xlabel('Year')  # X-axis label
plt.ylabel('GDP Growth (annual %)')  # Y-axis label
plt.legend()  # Add legend to distinguish historical vs. forecast
plt.grid()  # Add grid for better readability
plt.show()  # Display the plot

"""
Stationarity: ARIMA assumes a stationary series, so differencing (d) is critical. The ADF test guides this process.

AIC: Used to compare models; it penalizes complexity to avoid overfitting.

Forecast Horizon: 30 years is long for ARIMA, which may lead to flat predictions as it relies heavily on historical patterns.
"""


