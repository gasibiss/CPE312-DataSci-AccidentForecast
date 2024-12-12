import pandas as pd
import numpy as np
import os

try:
    # Get current working directory
    print("Current Working Directory:", os.getcwd())
    
    # Define base path dynamically
    base_path = os.path.join(os.getcwd(), "./data/")
    files = [os.path.join(base_path, f"accident{year}.xlsx") for year in range(2019, 2024)]
    
    df_list = []
    year_records = {}
    
    for file in files:
        if not os.path.exists(file):
            print(f"Warning: The file {file} was not found. Skipping this year.")
            continue
        
        df = pd.read_excel(file)
        year = file.split('accident')[1].split('.')[0]
        year_records[year] = len(df)
        df_list.append(df)
    
    if not df_list:
        raise FileNotFoundError("No valid accident files were found. Please check the paths.")
    
    df_combined = pd.concat(df_list, ignore_index=True)
    print("Record count for each year:")
    for year, count in year_records.items():
        print(f"Year {year}: {count} records")

    df_combined.columns = df_combined.columns.str.strip()
    columns_to_keep = ['วันที่เกิดเหตุ', 'เวลา', 'มูลเหตุสันนิษฐาน', 'ลักษณะการเกิดเหตุ', 'สภาพอากาศ', 
                       'ผู้เสียชีวิต', 'ผู้บาดเจ็บเล็กน้อย', 'ผู้บาดเจ็บสาหัส', 'รวมจำนวนผู้บาดเจ็บ']
    df_filtered = df_combined[columns_to_keep]

    count_null = 0
    for column in df_filtered.columns:
        if df_filtered[column].isnull().any():
            count_null += 1
            if df_filtered[column].dtype == 'object':
                df_filtered[column] = df_filtered[column].fillna('ข้อมูลสุ่ม')
            elif np.issubdtype(df_filtered[column].dtype, np.number):
                random_values = np.random.randint(1, 10, size=df_filtered[column].isnull().sum())
                df_filtered.loc[df_filtered[column].isnull(), column] = random_values

    df_filtered['รวมจำนวนผู้บาดเจ็บ'] = df_filtered['ผู้บาดเจ็บเล็กน้อย'] + df_filtered['ผู้บาดเจ็บสาหัส']
    df_filtered = df_filtered.drop(columns=['ผู้บาดเจ็บเล็กน้อย', 'ผู้บาดเจ็บสาหัส'])

    output_file = "./data/accident_combine_records_cleaned_final.xlsx"
    df_filtered.to_excel(output_file, index=False)
    print(f"Filtered file saved as {output_file}")
    print("Your empty data count -->", count_null)

except FileNotFoundError as e:
    print(f"File not found: {e}. Please ensure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
