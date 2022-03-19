# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'dbmaintain --db_host=localhost --db_port=27017 '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt

# %%
## */* RATES maintenance */* ##
from pymongo import MongoClient
from rata.marketon import get_data

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client['rates']

for db_col in db.list_collection_names():

    collection = db[db_col]
    mydoc = collection.find({})
    df = pd.DataFrame(mydoc)
    if len(df) == 0:
        print('No data in ', db_col)
    else:
        print('Cleaning ', db_col)
        #df['status'] = 'ok'
        df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume'], as_index=False).max()
        df = df[['query_tstamp', 'unix_tstamp', 'tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].sort_values('tstamp')
        
        collection.drop()
        db_bkp = client['rates']
        #db_bkp = client['rata' + '_rates']
        collection = db_bkp[db_col]
        collection.insert_many(df.to_dict(orient='records'))

client.close()
