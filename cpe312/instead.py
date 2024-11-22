import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the cleaned data
df = pd.read_excel('accident_combine_records_cleaned.xlsx')

# List the categorical columns you want to convert
categorical_columns = ['มูลเหตุสันนิษฐาน', 'ลักษณะการเกิดเหตุ', 'สภาพอากาศ']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Dictionary to store mappings
label_mappings = {}

# Loop through the categorical columns and convert them to numerical values
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    # Save the mapping for the current column
    label_mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Print the mappings for verification
for col, mapping in label_mappings.items():
    print(f"Mapping for {col}:")
    print(mapping)

# Save the transformed dataframe to a new Excel file
df.to_excel('accident_combined_numerical.xlsx', index=False)

# Save the mappings to a separate file
mappings_df = pd.DataFrame([
    {'Column': col, 'Text': text, 'Number': number}
    for col, mapping in label_mappings.items()
    for text, number in mapping.items()
])
mappings_df.to_excel('label_mappings.xlsx', index=False)

print("Transformed file saved as 'accident_combined_numerical.xlsx'")
print("Mappings saved as 'label_mappings.xlsx'")
