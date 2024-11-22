print("Hello world my EDA")

import pandas as pd
import matplotlib.pyplot as plt
import os
print("Current Working Directory:", os.getcwd())


# Load data from an Excel file
try:
    df = pd.read_excel('/cpe312/accident2019.xlsx')  # Replace with your actual file name and ensure the path is correct
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Display the first few rows of the dataframe
print(df.head())

# Ensure 'วันที่เกิดเหตุ' column is of datetime type
df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], errors='coerce')
df = df.dropna(subset=['วันที่เกิดเหตุ'])  # Drop rows with invalid dates

# Add 'ปี' (Year) and 'เดือน' (Month) columns for grouping
df['ปี'] = df['วันที่เกิดเหตุ'].dt.year
df['เดือน'] = df['วันที่เกิดเหตุ'].dt.month

# Count accidents by month
monthly_trend = df.groupby(['ปี', 'เดือน']).size().reset_index(name='จำนวนอุบัติเหตุ')

# Convert 'ปี' and 'เดือน' to a datetime format for plotting
monthly_trend['เดือน_ปี'] = pd.to_datetime(monthly_trend['ปี'].astype(str) + '-' + monthly_trend['เดือน'].astype(str))

# Plotting the monthly accident trend
plt.figure(figsize=(12, 6))
plt.plot(monthly_trend['เดือน_ปี'], monthly_trend['จำนวนอุบัติเหตุ'], marker='o', color='b', linestyle='-')
plt.title('แนวโน้มจำนวนอุบัติเหตุรายเดือน')
plt.xlabel('Month-Year')
plt.ylabel('Accident Amout')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
