import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace with your file path)
df = pd.read_excel('accident2023_test.xlsx')

# Convert the 'วันที่เกิดเหตุ' column to datetime (if not already)
df['วันที่เกิดเหตุ'] = pd.to_datetime(df['วันที่เกิดเหตุ'], format='%d/%m/%Y')

# Extract Month and Year from the date
df['Month'] = df['วันที่เกิดเหตุ'].dt.month
df['Year'] = df['วันที่เกิดเหตุ'].dt.year

# Group the data by month and year, summing the case amount (using 'รวมจำนวนผู้บาดเจ็บ' for the case count)
df_grouped = df.groupby(['Year', 'Month'])['รวมจำนวนผู้บาดเจ็บ'].sum().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
for year in df_grouped['Year'].unique():
    data_year = df_grouped[df_grouped['Year'] == year]
    plt.plot(data_year['Month'], data_year['รวมจำนวนผู้บาดเจ็บ'], label=f'Year {year}')

plt.xlabel('Month')
plt.ylabel('Case Amount (รวมจำนวนผู้บาดเจ็บ)')
plt.title('Accident Cases by Month')
plt.xticks(range(1, 13))  # Set x-axis to show months from 1 to 12
plt.legend(title='Year')
plt.grid(True)
plt.show()
