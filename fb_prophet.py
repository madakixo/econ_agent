# @madakixo 24032025-17:33
"""
Prophet model for forecasting with yearly trends (adjusted for annual data)
"""

# Import required libraries for data handling, visualization, and forecasting
import pandas as pd  # For data manipulation and DataFrame operations
import matplotlib.pyplot as plt  # For plotting the forecast and its components
from prophet import Prophet  # Prophet library for time series forecasting (formerly fbprophet)

# Load the dataset from a CSV file
# '1960_onwards1.csv' is expected to contain economic data starting from 1960
df = pd.read_csv('1960_onwards1.csv')

# Prepare data for Prophet
# Prophet requires a DataFrame with two columns: 'ds' (dates) and 'y' (values to forecast)
# Rename 'Year' to 'ds' and 'GDP growth (annual %)' to 'y' to match Prophet's expected format
prophet_df = df.rename(columns={'Year': 'ds', 'GDP growth (annual %)': 'y'})

# Convert the 'ds' column to datetime format
# Prophet requires dates in a proper datetime format; '%Y' indicates year-only (e.g., 1960)
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

# Define custom holidays or events specific to the context (e.g., Nigeria)
# Holidays can influence trends; here we add significant economic events
holidays = pd.DataFrame({
    'holiday': 'economic_event',  # Name of the holiday/event category
    'ds': pd.to_datetime(['1966-01-01', '1974-01-01']),  # Dates of events: 1966 coup, 1974 oil boom
    'lower_window': 0,  # Number of days before the event to include in its effect (0 = same day only)
    'upper_window': 1  # Number of days after the event to include (1 = effect lasts 1 day after)
})

# Initialize the Prophet model with specified parameters
model = Prophet(
    yearly_seasonality=True,  # Enable yearly seasonality to capture annual patterns
    weekly_seasonality=False,  # Disable weekly seasonality (irrelevant for yearly data)
    daily_seasonality=False,  # Disable daily seasonality (irrelevant for yearly data)
    seasonality_mode='additive',  # Use additive mode (default); assumes seasonality adds to the trend
    # Alternative: 'multiplicative' if seasonality scales with the trend (e.g., exponential growth)
    holidays=holidays  # Incorporate the custom holidays/events defined above
)

# Add country-specific holidays for Nigeria
# Prophet includes built-in holiday lists for various countries; 'NG' is the code for Nigeria
model.add_country_holidays(country_name='NG')

# Fit the Prophet model to the data
# The model learns trends, seasonality, and holiday effects from the historical data
model.fit(prophet_df)

# Create a future DataFrame for predictions
# Extends the historical dates by 30 years into the future for forecasting
future = model.make_future_dataframe(periods=30, freq='Y')  # 'Y' specifies yearly frequency
# 'future' includes both historical dates and the 30 future years

# Generate predictions using the fitted model
# Forecast includes trend, seasonality, holidays, and confidence intervals
forecast = model.predict(future)
# 'forecast' DataFrame contains columns like 'ds', 'yhat' (predicted value), 'yhat_lower', 'yhat_upper'

# Plot the forecast
# Prophet's built-in plot function shows historical data, forecast, and uncertainty intervals
fig1 = model.plot(forecast)
# Customize the plot with a title and axis labels
plt.title('Prophet Forecast for GDP Growth (Annual %) in Nigeria (1960–2054)')
plt.xlabel('Year')  # X-axis represents time
plt.ylabel('GDP Growth (annual %)')  # Y-axis represents the forecasted variable
plt.show()  # Display the forecast plot

# Plot the components of the forecast
# Breaks down the prediction into trend, yearly seasonality, and holiday effects
fig2 = model.plot_components(forecast)
plt.show()  # Display the components plot

# Display the last 5 forecasted values with confidence intervals
# Extract the final 5 rows (2050–2054) to show future predictions
print("\nLast 5 forecasted values (2050–2054):")
# Select relevant columns: date, predicted value, and lower/upper bounds of the confidence interval
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

"""
Prophet’s Strengths: Unlike ARIMA, Prophet handles missing data, trends, and seasonality automatically, making it user-friendly for annual data with holidays.

Seasonality: Yearly seasonality is enabled, but weekly/daily are disabled since the data is annual.

Holidays: Custom events and country-specific holidays (Nigeria) enhance the model’s ability to capture anomalies.

Additive vs. Multiplicative: Additive mode assumes seasonality effects are constant; switch to multiplicative if GDP growth scales exponentially.
"""
