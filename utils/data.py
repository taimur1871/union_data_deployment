# python 3.8

'''scripts to process the JSON from API'''

import pandas as pd

# create function to process data
def process_apicall(result):
    temp = result['source']['data']

    # get column names
    col_names = []
    for i in temp[0]:
        if type(temp[0][i]) is list:
            for j in temp[0][i]:
                if j['concept'] != 'Series':
                    col_names.append(j['concept'])
        else:
            col_names.append(i)
            
    df_temp = pd.DataFrame(columns=col_names)

    # append values to the new dataframe
    for i, data in enumerate(temp):
        temp_val = []
        #var = data['variable'][0]
        for j in data['variable']:
            if j['concept'] != 'Series':
                temp_val.append(j['value'])

        temp_val.append(data['value'])
        df_temp.loc[i] = temp_val

    return df_temp