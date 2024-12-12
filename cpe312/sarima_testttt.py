import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    logging.info(f"ADF Statistic: {result[0]}")
    logging.info(f"p-value: {result[1]}")
    logging.info("Critical Values:")
    for key, value in result[4].items():
        logging.info(f"\t{key}: {value}")

    if result[1] <= 0.05:
        logging.info("The series is stationary (p-value <= 0.05).")
        return True
    else:
        logging.info("The series is not stationary (p-value > 0.05).")
        return False

def prepare_data(df):
    df['Year'] = df['วันที่เกิดเหตุ'].dt.year
    df['Month'] = df['วันที่เกิดเหตุ'].dt.month
    monthly_accidents = df.groupby(['Year', 'Month']).size().reset_index(name='Accident_Count')
    monthly_accidents['Date'] = pd.to_datetime(monthly_accidents[['Year', 'Month']].assign(DAY=1))
    monthly_accidents.set_index('Date', inplace=True)
    return monthly_accidents.asfreq('MS', method='ffill')

def run_grid_search(train_data, test_data, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_period):
    parameter_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
    best_rmse = float('inf')
    best_order = None
    best_model = None

    for order in parameter_combinations:
        try:
            model = SARIMAX(train_data['Accident_Count'],
                            order=(order[0], order[1], order[2]),
                            seasonal_order=(order[3], order[4], order[5], seasonal_period))
            model_fit = model.fit()

            forecast = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
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

def evaluate_model(train_data, test_data, best_model, forecast_best):
    train_forecast = best_model.fittedvalues
    train_r2 = r2_score(train_data['Accident_Count'], train_forecast)

    test_r2 = r2_score(test_data['Accident_Count'], forecast_best)

    logging.info(f"Training R-squared: {train_r2}")
    logging.info(f"Testing R-squared: {test_r2}")

    return train_r2, test_r2

def check_overfitting(train_r2, test_r2, threshold=0.1):
    difference = train_r2 - test_r2
    if difference > threshold:
        logging.warning(f"Overfitting detected! Training R-squared: {train_r2:.4f}, Testing R-squared: {test_r2:.4f}, Difference: {difference:.4f}")
        return True
    else:
        logging.info(f"No significant overfitting. Training R-squared: {train_r2:.4f}, Testing R-squared: {test_r2:.4f}")
        return False

def plot_forecast(test_data, forecast_best, best_order):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Accident_Count'], label='Actual', color='blue')
    plt.plot(forecast_best.index, forecast_best, label='Forecast', color='red', linestyle='--')
    plt.title(f"SARIMA Forecast vs Actual Data - Best Model: {best_order}")
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

    # ทดสอบ Stationarity ของข้อมูลทั้งหมด
    if not test_stationarity(monthly_accidents['Accident_Count']):
        logging.info("Overall data is non-stationary, differencing may be required.")
    
    train_data = monthly_accidents[monthly_accidents.index < '2023-01-01']
    test_data = monthly_accidents[monthly_accidents.index >= '2023-01-01']

    # ทดสอบ Stationarity ของ Train และ Test
    logging.info("Checking stationarity of training data...")
    is_train_stationary = test_stationarity(train_data['Accident_Count'])
    if not is_train_stationary:
        logging.info("Train data is non-stationary. Differencing may be required.")

    logging.info("Checking stationarity of testing data...")
    is_test_stationary = test_stationarity(test_data['Accident_Count'])
    if not is_test_stationary:
        logging.info("Test data is non-stationary. Differencing may be required.")

    # Parameters for SARIMA
    p_values = [0, 1, 2, 3]
    d_values = [0, 1]
    q_values = [0, 1, 2, 3]
    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    seasonal_period = 12

    best_model, best_order, best_rmse = run_grid_search(train_data, test_data, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_period)

    logging.info(f"Best Model Order: {best_order}")
    logging.info(f"Best RMSE: {best_rmse}")

    forecast_best = best_model.predict(start=test_data.index[0], end=test_data.index[-1])
    forecast_best.index = test_data.index

    train_r2, test_r2 = evaluate_model(train_data, test_data, best_model, forecast_best)

    overfitting_detected = check_overfitting(train_r2, test_r2)

    if overfitting_detected:
        logging.info("Consider tuning the model parameters or simplifying the model to reduce overfitting.")
    else:
        logging.info("Model is well-balanced between training and testing data.")

    plot_forecast(test_data, forecast_best, best_order)

    forecast_df = pd.Data


except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
