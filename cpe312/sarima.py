import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from itertools import product

# Load the combined dataset
try:
    # Get current working directory
    print("Current Working Directory:", os.getcwd())
    
    # Define the file path
    file_path = os.path.join(os.getcwd(), 'accident_combine_records_cleaned_final.xlsx')
    
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Ensure columns are stripped of any extra spaces
    df.columns = df.columns.str.strip()

    # Check if required column exists
    required_columns = ['วันที่เกิดเหตุ']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Convert 'วันที่เกิดเหตุ' to datetime format
    df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], errors='coerce')

    # Handle missing dates
    df.dropna(subset=['วันที่เกิดเหตุ'], inplace=True)

    # Extract year and month
    df['Year'] = df['วันที่เกิดเหตุ'].dt.year
    df['Month'] = df['วันที่เกิดเหตุ'].dt.month

    # Count the number of accidents per month-year combination
    monthly_accidents = df.groupby(['Year', 'Month']).size().reset_index(name='Accident_Count')

    # Create a datetime column for the analysis
    monthly_accidents['Date'] = pd.to_datetime(monthly_accidents[['Year', 'Month']].assign(DAY=1))

    # Set the 'Date' as the index
    monthly_accidents.set_index('Date', inplace=True)

    # Check stationarity of the time series
    result = adfuller(monthly_accidents['Accident_Count'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("Data is non-stationary. Differencing may be required.")

    # Split data into training (2019-2021) and testing (2022)
    train_data = monthly_accidents.loc['2019-01-01':'2022-12-01']
    test_data = monthly_accidents.loc['2023-01-01':'2023-12-01']

    # Ensure train and test datasets are not empty
    if train_data.empty or test_data.empty:
        raise ValueError("Train or Test dataset is empty. Check data filtering conditions.")

    p_values = [0, 1, 2, 3]  # ลองเพิ่มค่าของ p
    d_values = [0, 1]  # ตรวจสอบว่าการต่างลำดับ 1 หรือไม่
    q_values = [0, 1, 2, 3]  # เพิ่มค่าของ q เพื่อจับความผันผวนในข้อมูล

    P_values = [0, 1, 2]  # เพิ่ม P เพื่อจับการเปลี่ยนแปลงตามฤดูกาล
    D_values = [0, 1]  # ทดสอบการต่างลำดับตามฤดูกาล
    Q_values = [0, 1, 2]  # เพิ่ม Q เพื่อจับความผิดพลาดตามฤดูกาล

    seasonal_period = 12  # หากข้อมูลมีฤดูกาลเป็นรายปี


    # Create a list of all possible combinations of (p, d, q, P, D, Q)
    parameter_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))

    # Store the results
    best_rmse = float('inf')
    best_order = None
    best_model = None

    # Grid search over different combinations of (p, d, q, P, D, Q)
    for order in parameter_combinations:
        try:
            # Fit a SARIMA model with the current combination of parameters
            model = SARIMAX(train_data['Accident_Count'],
                            order=(order[0], order[1], order[2]),
                            seasonal_order=(order[3], order[4], order[5], seasonal_period))
            model_fit = model.fit()

            # Forecast for the test period
            forecast = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
            forecast.index = test_data.index  # Align forecast index with test data

            # Calculate RMSE for this model
            mse = mean_squared_error(test_data['Accident_Count'], forecast)
            rmse = np.sqrt(mse)
            print(f'Order: {order}, RMSE: {rmse}')

            # If this is the best RMSE so far, store the model
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                best_model = model_fit

        except Exception as e:
            print(f"Error fitting model for order {order}: {e}")
            continue

    # Print the best model and RMSE
    print(f"Best Model Order: {best_order}")
    print(f"Best RMSE: {best_rmse}")

    # Plot the actual vs forecasted values for the test set using the best model
    forecast_best = best_model.predict(start=test_data.index[0], end=test_data.index[-1])
    forecast_best.index = test_data.index

    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['Accident_Count'], label='Actual', color='blue')
    plt.plot(forecast_best.index, forecast_best, label='Forecast', color='red', linestyle='--')
    plt.title(f"SARIMA Forecast vs Actual Data (2023) - Best Model: {best_order}")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.legend()
    plt.grid()
    plt.show()

    # Save the forecasted data to a new Excel file
    forecast_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual_Accident_Count': test_data['Accident_Count'].values,
        'Predicted_Accident_Count': forecast_best.values
    })
    forecast_df.to_excel('forecasted_accidents_2022_sarima.xlsx', index=False)

    print("Forecasting completed. Optimized forecasts saved in 'forecasted_accidents_2022_sarima.xlsx'.")

except ValueError as e:
    print(f"ValueError: {e}")
except KeyError as e:
    print(f"KeyError: {e}. Please check column names in the dataset.")
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
