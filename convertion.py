import pandas as pd

excel_file = 'dataset/CategoryCode_Mapping.xlsx'

# Read the Excel file into a dictionary of DataFrames, where keys are sheet names
excel_data = pd.read_excel(excel_file, sheet_name=None)

# Save each sheet as a separate CSV file
for sheet_name, df in excel_data.items():
    csv_filename = f'{sheet_name}.csv'
    df.to_csv(csv_filename, index=False)
