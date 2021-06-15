# python 3.8

'''this script provides a simple api call function for getting data \
from world bank data store, this is limited to population data'''

import requests
import pandas as pd
from utils.data import process_apicall

# setting global for dataset id to population
data_id = 40

def attr_df():
    '''
    function to get dataframe of attributes for each database
    This can be used to create a list of options, initially created 
    to explore more than one database
    '''
    # make rrequest to get a list of attributes
    url = 'http://api.worldbank.org/v2/sources/40/series?format=JSON'

    response = requests.get(url)
    attr_dict = response.json()

    # extract variables from dict
    attr_list = attr_dict['source'][0]['concept'][0]['variable']
    df_attrs = pd.DataFrame(attr_list)

    return df_attrs

def options():
    '''
    function to get list of attributes for each database
    also returns the dict which will be used to extract the code
    for the request
    '''
    # extract variables from dict
    url = 'http://api.worldbank.org/v2/sources/40/series?format=JSON'

    response = requests.get(url)
    temp_dict = response.json()
    attr_dicts = temp_dict['source'][0]['concept'][0]['variable']
    attr_list = [attr_dicts[i]['value'] for i in range(len(attr_dicts))]

    return attr_list, attr_dicts

def request_data(category):
    # build url for request from API
    url = 'http://api.worldbank.org/v2/sources/40/series/{}/time/all?per_page=100&page=1&format=JSON'.format(
            category)
    
    #return url
    response = requests.get(url)

    result = response.json()
    df = process_apicall(result)
    
    return df

def get_code(selected_option, option_dict):
    for i in option_dict:
        if i['value'] == selected_option:
            return i['id']

test = request_data('SP.POP.1519.FE')
