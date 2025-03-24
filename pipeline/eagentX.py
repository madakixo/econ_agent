#madakixo 24032024-18:16
#economic agent using xgboost not arima_forecast for better predictions
# Import necessary libraries for data handling, visualization, and modeling
import pandas as pd  # For data manipulation and DataFrame operations
import numpy as np  # For numerical computations and array handling
import plotly.express as px  # For creating interactive line plots
import plotly.graph_objects as go  # For building custom interactive visualizations
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model for time series forecasting
from statsmodels.tsa.stattools import adfuller  # Augmented Dickey-Fuller test for stationarity
from prophet import Prophet  # Prophet model for forecasting with trends and seasonality
from xgboost import XGBRegressor  # XGBoost regressor for advanced machine learning predictions
from sklearn.metrics import mean_squared_error  # To evaluate model performance with MSE
from itertools import product  # To generate combinations of ARIMA parameters (p, d, q)
import warnings  # To manage warning messages during execution
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Define the EconomicForecastAgent class to encapsulate forecasting functionality
class EconomicForecastAgent:
    def __init__(self, data_file):
        """
        Initialize the agent with a data file path and set up initial attributes.

        Args:
            data_file (str): Path to the CSV file containing economic data.
        """
        self.data_file = data_file  # Store the file path for loading data
        self.df = None  # Placeholder for the loaded DataFrame
        self.current_date = pd.to_datetime("2025-03-24")  # Set current date as context (March 24, 2025)

    def load_data(self):
        """
        Load and preprocess the CSV data into a pandas DataFrame.

        The method assumes the data contains a 'Year' column and economic indicators.
        Missing values are forward-filled to ensure continuity.
        """
        self.df = pd.read_csv(self.data_file)  # Load the CSV file into a DataFrame
        self.df['Year'] = pd.to_datetime(self.df['Year'], format='%Y')  # Convert 'Year' to datetime (year-only)
        self.df.set_index('Year', inplace=True)  # Set 'Year' as the index for time series operations
        self.df = self.df.fillna(method='ffill')  # Forward fill missing values to handle gaps
        print("Data loaded successfully. Columns available:", self.df.columns.tolist())  # Display available columns

    def test_stationarity(self, timeseries, label="series"):
        """
        Test if a time series is stationary using the Augmented Dickey-Fuller (ADF) test.

        Args:
            timeseries (pd.Series): The time series to test.
            label (str): Descriptive label for the series (e.g., 'original series').

        Returns:
            float: p-value from the ADF test.
        """
        result = adfuller(timeseries.dropna(), autolag='AIC')  # Perform ADF test, dropping NaNs
        print(f'\nStationarity test for {label}:')
        print('ADF Statistic: %f' % result[0])  # More negative = more evidence of stationarity
        print('p-value: %f' % result[1])  # p < 0.05 suggests stationarity (reject null)
        print('Critical Values:')  # Thresholds for significance levels
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))  # e.g., 1%, 5%, 10% critical values
        return result[1]  # Return p-value for further decision-making

    def determine_differencing(self, series, label="series"):
        """
        Determine the differencing order (d) required to make the series stationary for ARIMA.

        Args:
            series (pd.Series): The time series to analyze.
            label (str): Descriptive label for the series.

        Returns:
            tuple: (d, series_transformed) - differencing order and transformed series.
        """
        p_value = self.test_stationarity(series, f"original {label}")  # Test original series
        d = 0  # Initialize differencing order
        series_transformed = series.copy()  # Copy series to avoid modifying original
        if p_value >= 0.05:  # If non-stationary (fail to reject null hypothesis)
            print(f"\n{label} is non-stationary, applying first differencing...")
            series_transformed = series.diff().dropna()  # First difference to remove trend
            d = 1
            p_value = self.test_stationarity(series_transformed, f"first differenced {label}")
            if p_value >= 0.05:  # If still non-stationary
                print(f"\nStill non-stationary, applying second differencing...")
                series_transformed = series_transformed.diff().dropna()  # Second difference
                d = 2
                self.test_stationarity(series_transformed, f"second differenced {label}")
        return d, series_transformed  # Return differencing order and transformed series

    def arima_forecast(self, series, target_name, steps=30):
        """
        Fit an ARIMA model and generate a forecast.

        Args:
            series (pd.Series): The time series to forecast.
            target_name (str): Name of the target variable (e.g., 'GDP growth (annual %)').
            steps (int): Number of future periods to forecast (default: 30 years).

        Returns:
            pd.DataFrame: Forecasted values with dates as index.
        """
        d, _ = self.determine_differencing(series, target_name)  # Get differencing order
        p = q = range(0, 3)  # Define range for AR (p) and MA (q) parameters
        pdq = list(product(p, [d], q))  # Generate all (p, d, q) combinations

        best_aic = np.inf  # Initialize best AIC as infinity
        best_param = None  # Placeholder for best (p, d, q)
        best_model = None  # Placeholder for best fitted model

        # Grid search for best ARIMA parameters
        for param in pdq:
            try:
                mod = ARIMA(series, order=param)  # Initialize ARIMA with current parameters
                results = mod.fit()  # Fit the model
                if results.aic < best_aic:  # Compare AIC (lower is better)
                    best_aic = results.aic
                    best_param = param
                    best_model = results
                print(f'ARIMA{param} - AIC:{results.aic}')  # Log AIC for each attempt
            except Exception as e:
                print(f"ARIMA{param} failed: {e}")  # Log failures (e.g., convergence issues)
                continue

        # Fallback if no model converges
        if best_model is None:
            print("\nFalling back to ARIMA(1,1,1)...")
            best_param = (1, 1, 1)  # Default fallback parameters
            best_model = ARIMA(series, order=best_param).fit()

        print(f'\nBest ARIMA{best_param} - AIC:{best_aic}')  # Report best model
        forecast = best_model.forecast(steps=steps)  # Generate forecast
        forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.YearEnd(), periods=steps, freq='Y')  # Create future dates
        forecast_df = pd.DataFrame({target_name: forecast}, index=forecast_dates)  # Format forecast as DataFrame

        # Visualize with Plotly
        fig = go.Figure()  # Initialize interactive plot
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=f'Historical {target_name}', line=dict(color='blue')))  # Historical data
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[target_name], mode='lines', name=f'Forecasted {target_name}', line=dict(color='red', dash='dash')))  # Forecast
        fig.update_layout(title=f'{target_name} Forecast (ARIMA)', xaxis_title='Year', yaxis_title=target_name, template='plotly_white')  # Customize layout
        fig.show()  # Display plot

        return forecast_df  # Return forecast DataFrame

    def prophet_forecast(self, series, target_name, steps=30):
        """
        Fit a Prophet model and generate a forecast with trends and seasonality.

        Args:
            series (pd.Series): The time series to forecast.
            target_name (str): Name of the target variable.
            steps (int): Number of future periods to forecast (default: 30 years).

        Returns:
            pd.DataFrame: Forecasted values with confidence intervals.
        """
        prophet_df = pd.DataFrame({'ds': series.index, 'y': series.values})  # Format data for Prophet (ds: dates, y: values)
        holidays = pd.DataFrame({
            'holiday': 'economic_event',  # Define custom holiday category
            'ds': pd.to_datetime(['1966-01-01', '1974-01-01']),  # Significant events (e.g., Nigeria-specific)
            'lower_window': 0,  # Effect starts on event day
            'upper_window': 1  # Effect lasts 1 day after
        })

        # Initialize Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, holidays=holidays)  # Configure for yearly data
        model.add_country_holidays(country_name='NG')  # Add Nigerian holidays
        model.fit(prophet_df)  # Fit model to data

        future = model.make_future_dataframe(periods=steps, freq='Y')  # Create future dates (historical + forecast)
        forecast = model.predict(future)  # Generate forecast with trend, seasonality, and holidays

        # Visualize with Plotly
        fig = px.line(forecast, x='ds', y='yhat', title=f'Prophet Forecast for {target_name}', labels={'ds': 'Year', 'yhat': target_name})  # Plot forecast
        fig.add_scatter(x=series.index, y=series, mode='lines', name=f'Historical {target_name}', line=dict(color='blue'))  # Add historical data
        fig.update_traces(line=dict(dash='dash'), selector=dict(name='yhat'))  # Dash forecast line
        fig.update_layout(template='plotly_white')  # Use clean template
        fig.show()  # Display plot

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]  # Return forecast with confidence intervals

    def xgboost_forecast(self, series, target_name, steps=30):
        """
        Fit an XGBoost model with feature engineering and generate a forecast.

        Args:
            series (pd.Series): The time series to forecast.
            target_name (str): Name of the target variable.
            steps (int): Number of future periods to forecast (default: 30 years).

        Returns:
            pd.DataFrame: Forecasted values.
        """
        # Feature engineering: Create lag features and year for XGBoost
        data = pd.DataFrame({target_name: series})
        data['lag1'] = data[target_name].shift(1)  # Lag 1: previous year’s value
        data['lag2'] = data[target_name].shift(2)  # Lag 2: value two years prior
        data['year'] = data.index.year  # Extract year as a feature
        data = data.dropna()  # Remove rows with NaN due to shifting

        # Split data into training and testing sets (last 5 years for validation)
        train = data.iloc[:-5]  # All but last 5 years for training
        test = data.iloc[-5:]  # Last 5 years for testing

        X_train = train[['lag1', 'lag2', 'year']]  # Features for training
        y_train = train[target_name]  # Target for training
        X_test = test[['lag1', 'lag2', 'year']]  # Features for testing
        y_test = test[target_name]  # Target for testing

        # Initialize and fit XGBoost model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)  # Configure XGBoost
        model.fit(X_train, y_train)  # Train the model

        # Evaluate model performance on test set
        y_pred = model.predict(X_test)  # Predict on test data
        mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
        print(f"\nXGBoost MSE for {target_name}: {mse}")  # Report MSE

        # Generate future forecast
        future_data = pd.DataFrame(index=pd.date_range(start=series.index[-1] + pd.offsets.YearEnd(), periods=steps, freq='Y'))  # Future dates
        future_data['year'] = future_data.index.year  # Year feature
        future_data['lag1'] = series[-1]  # Initial lag1: last historical value
        future_data['lag2'] = series[-2]  # Initial lag2: second-to-last value

        predictions = []  # Store forecasted values
        for i in range(steps):  # Iteratively predict each future step
            X_future = future_data.iloc[i:i+1][['lag1', 'lag2', 'year']]  # Features for current step
            pred = model.predict(X_future)[0]  # Predict next value
            predictions.append(pred)  # Add to predictions
            if i < steps - 1:  # Update lags for next iteration
                future_data.iloc[i+1, future_data.columns.get_loc('lag2')] = future_data.iloc[i, future_data.columns.get_loc('lag1')]
                future_data.iloc[i+1, future_data.columns.get_loc('lag1')] = pred

        forecast_df = pd.DataFrame({target_name: predictions}, index=future_data.index)  # Format forecast

        # Visualize with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=f'Historical {target_name}', line=dict(color='blue')))  # Historical data
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[target_name], mode='lines', name=f'Forecasted {target_name}', line=dict(color='green', dash='dash')))  # Forecast
        fig.update_layout(title=f'{target_name} Forecast (XGBoost)', xaxis_title='Year', yaxis_title=target_name, template='plotly_white')  # Customize layout
        fig.show()  # Display plot

        return forecast_df  # Return forecast DataFrame

    def run_pipeline(self):
        """
        Execute the full forecasting pipeline for GDP growth and inflation using all models.
        """
        self.load_data()  # Load and preprocess data

        # Forecast GDP Growth with all models
        gdp_series = self.df['GDP growth (annual %)']  # Extract GDP growth series
        print("\n=== Forecasting GDP Growth with ARIMA ===")
        arima_gdp = self.arima_forecast(gdp_series, "GDP growth (annual %)")  # ARIMA forecast
        print("\nARIMA GDP Growth Forecast (2025–2034):")
        print(arima_gdp.loc['2025-01-01':'2034-01-01'])  # Show 2025-2034 subset

        print("\n=== Forecasting GDP Growth with Prophet ===")
        prophet_gdp = self.prophet_forecast(gdp_series, "GDP growth (annual %)")  # Prophet forecast
        print("\nProphet GDP Growth Forecast (2025–2034):")
        print(prophet_gdp.tail(10))  # Show last 10 years (includes 2025-2034)

        print("\n=== Forecasting GDP Growth with XGBoost ===")
        xgboost_gdp = self.xgboost_forecast(gdp_series, "GDP growth (annual %)")  # XGBoost forecast
        print("\nXGBoost GDP Growth Forecast (2025–2034):")
        print(xgboost_gdp.loc['2025-01-01':'2034-01-01'])  # Show 2025-2034 subset

        # Forecast Inflation with all models
        inflation_series = self.df['Inflation, consumer prices (annual %)']  # Extract inflation series
        print("\n=== Forecasting Inflation with ARIMA ===")
        arima_inflation = self.arima_forecast(inflation_series, "Inflation, consumer prices (annual %)")  # ARIMA forecast
        print("\nARIMA Inflation Forecast (2025–2034):")
        print(arima_inflation.loc['2025-01-01':'2034-01-01'])  # Show 2025-2034 subset

        print("\n=== Forecasting Inflation with Prophet ===")
        prophet_inflation = self.prophet_forecast(inflation_series, "Inflation, consumer prices (annual %)")  # Prophet forecast
        print("\nProphet Inflation Forecast (2025–2034):")
        print(prophet_inflation.tail(10))  # Show last 10 years (includes 2025-2034)

        print("\n=== Forecasting Inflation with XGBoost ===")
        xgboost_inflation = self.xgboost_forecast(inflation_series, "Inflation, consumer prices (annual %)")  # XGBoost forecast
        print("\nXGBoost Inflation Forecast (2025–2034):")
        print(xgboost_inflation.loc['2025-01-01':'2034-01-01'])  # Show 2025-2034 subset

# Entry point to run the script
if __name__ == "__main__":
    agent = EconomicForecastAgent('1960_onwards1.csv')  # Initialize agent with data file
    agent.run_pipeline()  # Execute the full forecasting pipeline
