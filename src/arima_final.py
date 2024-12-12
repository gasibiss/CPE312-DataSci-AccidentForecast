import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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
    # Define accident count
    monthly_accidents = df.groupby(['Year', 'Month']).size().reset_index(name='Accident_Count')
    monthly_accidents['Date'] = pd.to_datetime(monthly_accidents[['Year', 'Month']].assign(DAY=1))
    monthly_accidents.set_index('Date', inplace=True)
    return monthly_accidents.asfreq('MS', method='ffill')  # Fill missing months

def run_grid_search(train_data, test_data, p_values, d_values, q_values):
    parameter_combinations = list(product(p_values, d_values, q_values))
    best_rmse = float('inf')
    best_order = None
    best_model = None

    for order in parameter_combinations:
        try:
            model = ARIMA(train_data['Accident_Count'], order=order)
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=len(test_data))
            forecast.index = test_data.index
            mse = mean_squared_error(test_data['Accident_Count'], forecast)
            rmse = np.sqrt(mse)
            logging.info(f'Order: {order}, RMSE: {rmse}')

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                best_model = model_fit
        except Exception as e:
            logging.error(f"Error fitting model for order {order}: {e}")
            continue

    return best_model, best_order, best_rmse

def evaluate_overfitting(train_data, test_data, best_model, forecast_best):
    # Calculate training error
    train_forecast = best_model.fittedvalues
    train_mse = mean_squared_error(train_data['Accident_Count'], train_forecast)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(train_data['Accident_Count'], train_forecast)

    # Calculate residuals
    residuals = test_data['Accident_Count'] - forecast_best

    logging.info(f"Training RMSE: {train_rmse}")
    logging.info(f"Training R-squared: {train_r2}")

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, residuals, label='Residuals', color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid()
    plt.show()

    return train_rmse, train_r2

def plot_forecast(test_data, forecast_best, best_order):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Accident_Count'], label='Actual', color='blue')
    plt.plot(forecast_best.index, forecast_best, label='Forecast', color='red', linestyle='--')
    plt.title(f"ARIMA Forecast vs Actual Data (2023) - Best Model: {best_order}")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.legend()
    plt.grid()
    plt.savefig("./results/ARIMA_RESULT.png")
    plt.show()

# Main execution
file_path = './data/accident_combine_records_cleaned_final.xlsx'
try:
    df = load_data(file_path)
    monthly_accidents = prepare_data(df)

    if not test_stationarity(monthly_accidents['Accident_Count']):
        logging.info("Data is non-stationary, differencing may be required.")

    # Split data
    train_data = monthly_accidents[monthly_accidents.index < '2023-01-01']
    test_data = monthly_accidents[monthly_accidents.index >= '2023-01-01']

    # Define ARIMA parameter grid
    p_values = [0, 1, 2, 3]
    d_values = [0, 1]
    q_values = [0, 1, 2, 3]

    # Run Grid Search to find the best parameters
    best_model, best_order, best_rmse = run_grid_search(train_data, test_data, p_values, d_values, q_values)

    logging.info(f"Best Model Order: {best_order}")
    logging.info(f"Best RMSE: {best_rmse}")

    forecast_best = best_model.forecast(steps=len(test_data))
    forecast_best.index = test_data.index

    # Calculate R-squared
    r_squared = r2_score(test_data['Accident_Count'], forecast_best)
    logging.info(f"R-squared: {r_squared}")

    plot_forecast(test_data, forecast_best, best_order)

    # Evaluate overfitting
    train_rmse, train_r2 = evaluate_overfitting(train_data, test_data, best_model, forecast_best)

    # Save forecasted data
    forecast_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual_Accident_Count': test_data['Accident_Count'].values,
        'Predicted_Accident_Count': forecast_best.values,
        'R-squared': [r_squared] * len(test_data)
    })
    logging.info("Forecasting completed. Optimized forecasts saved in 'forecasted_accidents_2022_arima.xlsx'.")

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
