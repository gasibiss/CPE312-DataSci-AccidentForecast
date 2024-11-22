import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

# Load the combined dataset
try:
    # Load the dataset (make sure to provide the correct path)
    df = pd.read_excel('accident_combine_records_cleaned.xlsx')
    
    # Ensure columns are stripped of any extra spaces
    df.columns = df.columns.str.strip()

    # Convert 'วันที่เกิดเหตุ' to datetime format
    df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], errors='coerce')

    # Extract year and month
    df['Year'] = df['วันที่เกิดเหตุ'].dt.year
    df['Month'] = df['วันที่เกิดเหตุ'].dt.month

    # Count the number of accidents per month-year combination
    monthly_accidents = df.groupby(['Year', 'Month']).size().reset_index(name='Accident_Count')

    # Create a datetime column for plotting
    monthly_accidents['Date'] = pd.to_datetime(monthly_accidents[['Year', 'Month']].assign(DAY=1))

    # Set the 'Date' as the index
    monthly_accidents.set_index('Date', inplace=True)

    # Plot the accident count to visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_accidents['Accident_Count'])
    plt.title("Monthly Accident Counts")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.show()

    # Split data into training and testing sets (80% train, 20% test)
    train = monthly_accidents[['Accident_Count']]

    print(train)

    # Fit an ARIMA model (you may need to adjust the order depending on your data)
    model = ARIMA(train['Accident_Count'], order=(5, 1, 0))  # p=5, d=1, q=0 as an example
    model_fit = model.fit()

    # Summary of the ARIMA model
    print(model_fit.summary())

    # Forecast the future values (next 12 months, for the next year)
    forecast_steps = 12  # Forecasting for the next 12 months (next year)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate months for the next year (1 to 12)
    forecast_months = np.arange(1, forecast_steps + 1)

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_months, forecast, label='Forecast', color='red')
    plt.title("ARIMA Forecast for the Next Year (12 Months)")
    plt.xlabel("Month")
    plt.ylabel("Predicted Accident Count")
    plt.xticks(forecast_months)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the forecasted data to a new Excel file (optional)
    forecast_df = pd.DataFrame({
        'Month': forecast_months,
        'Predicted_Accident_Count': forecast
    })
    forecast_df.to_excel('forecasted_accidents_next_year.xlsx', index=False)

    print("Forecasting completed. Forecasts saved in 'forecasted_accidents_next_year.xlsx'.")

except KeyError as e:
    print(f"KeyError: {e}. Please check column names in the dataset.")
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
