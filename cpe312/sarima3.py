import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()  # Strip spaces
    df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], errors='coerce')
    df.dropna(subset=['วันที่เกิดเหตุ'], inplace=True)
    return df

def test_stationarity(series):
    result = adfuller(series)
    logging.info(f'ADF Statistic: {result[0]}')
    logging.info(f'p-value: {result[1]}')
    return result[1] <= 0.05

def prepare_data(df):
    df['Year'] = df['วันที่เกิดเหตุ'].dt.year
    df['Month'] = df['วันที่เกิดเหตุ'].dt.month
    monthly_accidents = df.groupby(['Year', 'Month']).size().reset_index(name='Accident_Count')
    monthly_accidents['Date'] = pd.to_datetime(monthly_accidents[['Year', 'Month']].assign(DAY=1))
    monthly_accidents.set_index('Date', inplace=True)
    return monthly_accidents.asfreq('MS', method='ffill')  # Fill missing months

def cross_validate_sarima(data, order, seasonal_period, splits=5):
    """Perform cross-validation for a specific SARIMA order."""
    tscv = TimeSeriesSplit(n_splits=splits)
    rmse_scores = []

    for train_index, test_index in tscv.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        try:
            model = SARIMAX(train,
                            order=(order[0], order[1], order[2]),
                            seasonal_order=(order[3], order[4], order[5], seasonal_period))
            model_fit = model.fit(disp=False)

            forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
            rmse = np.sqrt(mean_squared_error(test, forecast))
            rmse_scores.append(rmse)
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")
            return float('inf')  # Return a very high error for failed models

    return np.mean(rmse_scores)  # Return the average RMSE across splits

def run_grid_search_with_cv(data, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_period):
    """Run grid search with cross-validation."""
    parameter_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
    best_rmse = float('inf')
    best_order = None

    for order in parameter_combinations:
        try:
            logging.info(f"Evaluating order: {order}")
            rmse = cross_validate_sarima(data['Accident_Count'], order, seasonal_period)
            logging.info(f"Order: {order}, Cross-Validated RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception as e:
            logging.error(f"Error evaluating order {order}: {e}")
            continue

    return best_order, best_rmse

def plot_forecast(test_data, forecast_best, best_order):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Accident_Count'], label='Actual', color='blue')
    plt.plot(forecast_best.index, forecast_best, label='Forecast', color='red', linestyle='--')
    plt.title(f"SARIMA Forecast vs Actual Data (2023) - Best Model: {best_order}")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
file_path = 'accident_combine_records_cleaned_final.xlsx'

try:
    df = load_data(file_path)
    monthly_accidents = prepare_data(df)

    if not test_stationarity(monthly_accidents['Accident_Count']):
        logging.info("Data is non-stationary, differencing may be required.")
    
    # Split data
    train_data = monthly_accidents[monthly_accidents.index < '2023-01-01']
    test_data = monthly_accidents[monthly_accidents.index >= '2023-01-01']

    # Define SARIMA parameter grid
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    seasonal_period = 12

    # Run grid search with cross-validation
    best_order, best_rmse = run_grid_search_with_cv(
        monthly_accidents, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_period
    )

    logging.info(f"Best Model Order (with CV): {best_order}")
    logging.info(f"Best Cross-Validated RMSE: {best_rmse}")

    # Fit and forecast with the best model
    best_model = SARIMAX(train_data['Accident_Count'],
                         order=(best_order[0], best_order[1], best_order[2]),
                         seasonal_order=(best_order[3], best_order[4], best_order[5], seasonal_period)).fit()

    forecast_best = best_model.predict(start=test_data.index[0], end=test_data.index[-1])
    plot_forecast(test_data, forecast_best, best_order)

    # Save forecasted data
    forecast_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual_Accident_Count': test_data['Accident_Count'].values,
        'Predicted_Accident_Count': forecast_best.values
    })
    forecast_df.to_excel('forecasted_accidents_2022_sarima_with_cv.xlsx', index=False)

    logging.info("Forecasting completed. Optimized forecasts saved in 'forecasted_accidents_2022_sarima_with_cv.xlsx'.")

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
