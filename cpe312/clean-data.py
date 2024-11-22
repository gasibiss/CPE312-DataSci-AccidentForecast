import pandas as pd
import numpy as np
import os  # Importing os module to check for file existence

try:
    # Load the Excel files for each year
    files = ['/cpe312/accident2019.xlsx', '/cpe312/accident2020.xlsx', '/cpe312/accident2021.xlsx', '/cpe312/accident2022.xlsx']
    
    # Initialize an empty list to hold dataframes for each year
    df_list = []
    year_records = {}  # To store the count of records per year
    
    # Loop through each file and load the data
    for file in files:
        # Check if the file exists
        if not os.path.exists(file):
            print(f"Warning: The file {file} was not found. Skipping this year.")
            continue  # Skip this file if it's not found
        
        # Read the data from the current year
        df = pd.read_excel(file)
        
        # Extract the year from the filename or manually set it if necessary
        year = file.split('accident')[1].split('.')[0]  # Extract year from the filename, e.g., 2019, 2020, 2021
        year_records[year] = len(df)  # Store the number of records for the current year
        
        # Append the dataframe to the list
        df_list.append(df)
    
    if not df_list:
        raise FileNotFoundError("No valid accident files were found. Please check the paths.")
    
    # Combine all data into one dataframe
    df_combined = pd.concat(df_list, ignore_index=True)

    # Print the record counts for each year
    print("Record count for each year:")
    for year, count in year_records.items():
        print(f"Year {year}: {count} records")

    # Proceed with your data processing and cleaning as before
    df_combined.columns = df_combined.columns.str.strip()

    # Select only the specific columns to keep
    columns_to_keep = ['วันที่เกิดเหตุ', 'เวลา', 'มูลเหตุสันนิษฐาน', 'ลักษณะการเกิดเหตุ', 'สภาพอากาศ', 
                       'ผู้เสียชีวิต', 'ผู้บาดเจ็บเล็กน้อย', 'ผู้บาดเจ็บสาหัส', 'รวมจำนวนผู้บาดเจ็บ']
    df_filtered = df_combined[columns_to_keep]

    count_null = 0

    # Fill empty data with random values
    for column in df_filtered.columns:
        if df_filtered[column].isnull().any():
            count_null += 1
            if df_filtered[column].dtype == 'object':  # For categorical or text data
                df_filtered[column] = df_filtered[column].fillna('ข้อมูลสุ่ม')
            elif np.issubdtype(df_filtered[column].dtype, np.number):  # For numeric data
                random_values = np.random.randint(1, 10, size=df_filtered[column].isnull().sum())
                df_filtered.loc[df_filtered[column].isnull(), column] = random_values

    # Combine 'ผู้บาดเจ็บเล็กน้อย' and 'ผู้บาดเจ็บสาหัส' into 'รวมจำนวนผู้บาดเจ็บ'
    df_filtered['รวมจำนวนผู้บาดเจ็บ'] = df_filtered['ผู้บาดเจ็บเล็กน้อย'] + df_filtered['ผู้บาดเจ็บสาหัส']

    # Drop the original columns 'ผู้บาดเจ็บเล็กน้อย' and 'ผู้บาดเจ็บสาหัส'
    df_filtered = df_filtered.drop(columns=['ผู้บาดเจ็บเล็กน้อย', 'ผู้บาดเจ็บสาหัส'])

    # Save the cleaned DataFrame to a new Excel file
    output_file = "accident_combine_records_cleaned.xlsx"
    df_filtered.to_excel(output_file, index=False)
    print(f"Filtered file saved as {output_file}")
    print("Your empty data count -->", count_null)

except KeyError as e:
    print(f"KeyError: {e}. Please check column names in the dataset.")
except FileNotFoundError as e:
    print(f"File not found: {e}. Please ensure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
