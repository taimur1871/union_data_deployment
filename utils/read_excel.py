import base64
import datetime
import io

import pandas as pd

# function to read excel files to pandas
# may replace it with dash or pyspark in future 
# if dealing with larger datasets
def parse_contents(file_path, filename):
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(file_path)
            df_head = df.head()
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(file_path)
            df_head = df.head()
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(file_path)
            df_head = df.head()

    except Exception as e:
        print(e)
        return ('There was an error processing this file.')

    return df, df_head