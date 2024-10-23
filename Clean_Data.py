import os
import pandas as pd

# Directory containing the CSV files
directory = 'Data-Refined\\2024'
#Data-Refined

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Replace 'N/A' with NaN
        df.replace('N/A', pd.NA, inplace=True)
        
        # Drop rows with any NaN values
        df.dropna(inplace=True)

        # Remove specified columns if they exist in the DataFrame
        columns_to_remove = ['Player','Date' ,'Position', 'TM', 'OPP','AST','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','TOV','TS%','USG%','eFG%','ORtg','DRtg','GmSc','BPM']
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], axis=1, inplace=True)
        
        # Save the cleaned data to a new CSV file, overwriting the original
        df.to_csv(file_path, index=False)

print("All files have been processed and cleaned.")
