Economic Forecasting Scripts: ARIMA and Prophet Models
This repository contains two Python scripts for forecasting economic indicators, specifically GDP growth and inflation, using the ARIMA and Prophet models. The scripts are designed to work with annual data from the dataset '1960_onwards1.csv', which includes historical economic data starting from 1960.
ARIMA Script: Focuses on forecasting GDP growth using the ARIMA model, with steps for stationarity testing, differencing, and parameter tuning.

Prophet Script: Uses the Prophet model to forecast GDP growth, incorporating yearly trends, custom holidays, and country-specific holidays for Nigeria.

Both scripts generate forecasts for the next 30 years and provide visualizations of historical and forecasted data.
Table of Contents
Introduction (#introduction)

Scripts Overview (#scripts-overview)

Requirements (#requirements)

Usage Instructions (#usage-instructions)

Model Explanations (#model-explanations)

Customization (#customization)

Output (#output)

Notes (#notes)

Introduction
The purpose of these scripts is to forecast key economic indicators—specifically, GDP growth and inflation—using two different time series forecasting models: ARIMA and Prophet. 
ARIMA (AutoRegressive Integrated Moving Average) is a traditional time series model that requires stationarity and involves tuning parameters for autoregression, differencing, and moving averages.

Prophet is a modern forecasting tool developed by Facebook (now Meta), designed to handle time series data with trends, seasonality, and holiday effects, making it particularly user-friendly for annual data with known events.

The scripts are built to work with the dataset '1960_onwards1.csv', which contains annual economic data starting from 1960. The ARIMA script is tailored for GDP growth forecasting, while the Prophet script can be easily adapted for other indicators like inflation.
Scripts Overview
1. ARIMA Script
Purpose: Forecasts GDP growth using the ARIMA model.

Key Steps:
Stationarity Testing: Uses the Augmented Dickey-Fuller (ADF) test to check if the time series is stationary. If not, differencing is applied.

Parameter Tuning: Performs a grid search over possible values of p (autoregressive terms), d (differencing order), and q (moving average terms) to find the best model based on the Akaike Information Criterion (AIC).

Forecasting: Generates a 30-year forecast using the best ARIMA model.

Visualization: Plots historical GDP growth and the forecasted values.

2. Prophet Script
Purpose: Forecasts GDP growth using the Prophet model, with support for yearly trends and holiday effects.

Key Features:
Data Preparation: Renames columns to meet Prophet’s requirements (ds for dates, y for the target variable).

Custom Holidays: Includes significant economic events (e.g., 1966 coup, 1974 oil boom) and Nigerian country-specific holidays.

Model Configuration: Enables yearly seasonality and disables weekly/daily seasonality (since the data is annual).

Forecasting: Generates a 30-year forecast, including trend, seasonality, and holiday effects.

Visualization: Plots the forecast and its components (trend, seasonality, holidays).

Requirements
To run these scripts, you need the following:
Python 3.x

Libraries:
pandas: For data manipulation.

numpy: For numerical operations.

matplotlib: For plotting.

statsmodels: For ARIMA modeling and stationarity testing.

prophet: For the Prophet forecasting model.

Dataset: '1960_onwards1.csv', which should contain at least the columns 'Year' and 'GDP growth (annual %)'. For inflation forecasting, ensure the dataset includes 'Inflation, consumer prices (annual %)'.

Install the required libraries using:
bash

pip install pandas numpy matplotlib statsmodels prophet

Usage Instructions
Prepare the Dataset:
Ensure '1960_onwards1.csv' is in the same directory as the scripts or provide the correct path.

Run the ARIMA Script:
The ARIMA script is set to forecast GDP growth by default.

To run: python arima_forecast.py (assuming the script is saved as arima_forecast.py).

Run the Prophet Script:
The Prophet script is also set to forecast GDP growth but can be adapted for other indicators.

To run: python prophet_forecast.py (assuming the script is saved as prophet_forecast.py).

Adapt for Inflation:
To forecast inflation, modify the target_col in the ARIMA script or the column renaming in the Prophet script to use 'Inflation, consumer prices (annual %)'.

Model Explanations
ARIMA (AutoRegressive Integrated Moving Average)
Overview: ARIMA is a widely used model for time series forecasting. It combines autoregression (AR), differencing (I), and moving averages (MA).

Parameters:
p: Number of autoregressive terms (lags of the dependent variable).

d: Order of differencing to make the series stationary.

q: Number of moving average terms (lags of the forecast errors).

Stationarity: ARIMA requires the time series to be stationary. The script tests for stationarity using the ADF test and applies differencing if necessary.

Parameter Tuning: A grid search is performed over p, d, and q to select the model with the lowest AIC.

Prophet
Overview: Prophet is a forecasting tool designed for time series data with strong seasonal patterns and known holidays. 
It is particularly effective for data with missing values or outliers.

Key Features:
Automatically detects trends and seasonality.

Allows for the inclusion of custom holidays and events.

Handles annual data effectively with yearly seasonality.

Configuration: The script enables yearly seasonality and disables weekly/daily seasonality. It also incorporates Nigerian holidays and custom economic events.

Forecasting: Prophet generates forecasts with confidence intervals and decomposes the forecast into trend, seasonality, and holiday components.

Customization
Change Target Variable: To forecast a different indicator (e.g., inflation), update the target_col in the ARIMA script or modify the column renaming in the Prophet script.

Adjust Forecast Horizon: Change the steps parameter in the ARIMA forecast or the periods in Prophet’s make_future_dataframe to adjust the forecast length.

Add More Holidays: Extend the holidays DataFrame in the Prophet script to include additional significant events.

Model Parameters: For ARIMA, expand the range of p and q in the grid search for potentially better models. 
For Prophet, experiment with seasonality_mode='multiplicative' if the data shows exponential growth.

Output
Forecasts: Both scripts output a 30-year forecast for the target variable.
ARIMA: Prints the forecasted values directly.

Prophet: Prints the last 5 forecasted values with confidence intervals.

Visualizations:
ARIMA: A plot showing historical GDP growth and the forecasted values.

Prophet: Two plots:
Forecast plot: Historical data, forecasted values, and uncertainty intervals.

Components plot: Breakdown of the forecast into trend, yearly seasonality, and holiday effects.

Notes
Stationarity in ARIMA: The ARIMA model assumes the time series is stationary. The script handles this by testing and applying differencing as needed.

Annual Data in Prophet: Prophet is configured to handle annual data by enabling yearly seasonality and disabling weekly/daily seasonality.

Long-Term Forecasts: Forecasting 30 years ahead may lead to less reliable predictions, especially with ARIMA, as it relies heavily on historical patterns. 
Prophet may handle long-term trends better due to its trend modeling.

Data Assumptions: The scripts assume the dataset contains no missing values or handles them via forward-filling. Ensure your data is clean or adjust the scripts accordingly.

