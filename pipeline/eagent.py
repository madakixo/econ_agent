import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from itertools import product
import warnings
warnings.filterwarnings("ignore")

class EconomicForecastAgent:
    def __init__(self, data_file):
        """Initialize the agent with the data file."""
        self.data_file = data_file
        self.df = None
        self.current_date = pd.to_datetime("2025-03-24")  # Current date as per context

    def load_data(self):
        """Load and preprocess the data."""
        self.df = pd.read_csv(self.data_file)
        self.df['Year'] = pd.to_datetime(self.df['Year'], format='%Y')
        self.df.set_index('Year', inplace=True)
        print("Data loaded successfully. Columns available:", self.df.columns.tolist())

    def test_stationarity(self, timeseries, label="series"):
        """Test stationarity using Augmented Dickey-Fuller test."""
        result = adfuller(timeseries.dropna(), autolag='AIC')
        print(f'\nStationarity test for {label}:')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        return result[1]  # Return p-value

    def determine_differencing(self, series, label="series"):
        """Determine the required differencing order for ARIMA."""
        p_value = self.test_stationarity(series, f"original {label}")
        d = 0
        series_transformed = series.copy()
        if p_value >= 0.05:
            print(f"\n{label} is non-stationary, applying first differencing...")
            series_transformed = series.diff().dropna()
            d = 1
            p_value = self.test_stationarity(series_transformed, f"first differenced {label}")
            if p_value >= 0.05:
                print(f"\nStill non-stationary, applying second differencing...")
                series_transformed = series_transformed.diff().dropna()
                d = 2
                self.test_stationarity(series_transformed, f"second differenced {label}")
        return d, series_transformed

    def arima_forecast(self, series, target_name, steps=30):
        """Fit ARIMA model and forecast."""
        # Determine differencing
        d, series_transformed = self.determine_differencing(series, target_name)

        # Grid search for ARIMA parameters
        p = q = range(0, 3)
        d_range = [d]
        pdq = list(product(p, d_range, q))

        best_aic = np.inf
        best_param = None
        best_model = None

        for param in pdq:
            try:
                mod = ARIMA(series, order=param)
                results = mod.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_param = param
                    best_model = results
                print(f'ARIMA{param} - AIC:{results.aic}')
            except Exception as e:
                print(f"ARIMA{param} failed: {e}")
                continue

        if best_model is None:
            print("\nAll ARIMA models failed. Using fallback ARIMA(1,1,1)...")
            best_param = (1, 1, 1)
            best_model = ARIMA(series, order=best_param).fit()

        print(f'\nBest ARIMA{best_param} - AIC:{best_aic}')

        # Forecast
        forecast = best_model.forecast(steps=steps)
        forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.YearEnd(), periods=steps, freq='Y')
        forecast_df = pd.DataFrame({target_name: forecast}, index=forecast_dates)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(series, label=f'Historical {target_name}', color='blue')
        plt.plot(forecast_df, label=f'Forecasted {target_name}', color='red', linestyle='--')
        plt.title(f'{target_name} Forecast')
        plt.xlabel('Year')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid()
        plt.show()

        return forecast_df

    def prophet_forecast(self, series, target_name, steps=30):
        """Fit Prophet model and forecast."""
        prophet_df = pd.DataFrame({'ds': series.index, 'y': series.values})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        # Define holidays (example for Nigeria)
        holidays = pd.DataFrame({
            'holiday': 'economic_event',
            'ds': pd.to_datetime(['1966-01-01', '1974-01-01']),
            'lower_window': 0,
            'upper_window': 1
        })

        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays
        )
        model.add_country_holidays(country_name='NG')
        model.fit(prophet_df)

        # Future dataframe
        future = model.make_future_dataframe(periods=steps, freq='Y')
        forecast = model.predict(future)

        # Plot
        fig1 = model.plot(forecast)
        plt.title(f'Prophet Forecast for {target_name}')
        plt.xlabel('Year')
        plt.ylabel(target_name)
        plt.show()

        fig2 = model.plot_components(forecast)
        plt.show()

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def run_pipeline(self):
        """Run the full forecasting pipeline."""
        self.load_data()

        # Forecast GDP Growth
        gdp_series = self.df['GDP growth (annual %)'].fillna(method='ffill')
        print("\n=== Forecasting GDP Growth with ARIMA ===")
        arima_gdp_forecast = self.arima_forecast(gdp_series, "GDP growth (annual %)")
        print("\nARIMA GDP Growth Forecast (2025–2034):")
        print(arima_gdp_forecast.loc['2025-01-01':'2034-01-01'])

        print("\n=== Forecasting GDP Growth with Prophet ===")
        prophet_gdp_forecast = self.prophet_forecast(gdp_series, "GDP growth (annual %)")
        print("\nProphet GDP Growth Forecast (2025–2034):")
        print(prophet_gdp_forecast.tail(10))

        # Forecast Inflation
        inflation_series = self.df['Inflation, consumer prices (annual %)'].fillna(method='ffill')
        print("\n=== Forecasting Inflation with ARIMA ===")
        arima_inflation_forecast = self.arima_forecast(inflation_series, "Inflation, consumer prices (annual %)")
        print("\nARIMA Inflation Forecast (2025–2034):")
        print(arima_inflation_forecast.loc['2025-01-01':'2034-01-01'])

        print("\n=== Forecasting Inflation with Prophet ===")
        prophet_inflation_forecast = self.prophet_forecast(inflation_series, "Inflation, consumer prices (annual %)")
        print("\nProphet Inflation Forecast (2025–2034):")
        print(prophet_inflation_forecast.tail(10))

if __name__ == "__main__":
    agent = EconomicForecastAgent('1960_onwards1.csv')
    agent.run_pipeline()
