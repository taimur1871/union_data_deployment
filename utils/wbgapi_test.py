# python 3.8

'this is an example of code for gathering data from world bank open source databank'

import wbgapi as wb

dbs = wb.source.info()
db_list = dbs.table()

# set db to population
wb.db = 40

# get list of population features
features = wb.source.info()

# use to check how many seires options are available
# for this project sticking to 40 Population estimates and projections
'''
for i in range(len(db_list)):
    wb.db = i
    try:
        t = wb.series.info()
        print(db_list[i][0], db_list[i][1], len(t.table()))
    except wb.APIError:
        continue
        #print(i, 'No series info found')
'''

# function to get the details of the dataset
def data_properties(keyword):
    db_list.index(keyword)
    wb.db = 40

