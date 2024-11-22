import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the cleaned data
df = pd.read_excel('accident_combine_records_cleaned.xlsx')

# List the categorical columns you want to convert
categorical_columns = ['มูลเหตุสันนิษฐาน', 'ลักษณะการเกิดเหตุ', 'สภาพอากาศ']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Loop through the categorical columns and convert them to numerical values
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Print the first few rows to verify the transformation
print(df.head())

# Save the transformed dataframe to a new Excel file
df.to_excel('accident_combined_numerical.xlsx', index=False)

print("Transformed file saved as 'accident_combined_numerical.xlsx'")
