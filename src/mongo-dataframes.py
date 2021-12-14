# %%
# Global imports
import pandas as pd
import datetime as dt
import numpy as np
import pickle
import gzip
from pymongo import MongoClient
from datetime import datetime

# %%
_conf = {'db_host': 'localhost',
         'db_port': 27017,
         'dbs_prefix': 'rata_test',
         'interval': 5
}

symbols = [
        'AUDUSD',
        'USDJPY',
        'AUDCHF',
        'USDCAD',
        'GBPAUD',
        'EURUSD',
        'EURGBP',
        'USDCHF',
        'GBPNZD']

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_classifiers']

df = pd.DataFrame()
for s in symbols:
    _conf['symbol'] = s
    db_col = _conf['symbol'] + '_' + str(_conf['interval'])
    collection = db[db_col]
    mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 5000) #
    df_symbol = pd.DataFrame(mydoc)
    df = df.append(df_symbol)
df
# %%
df[df['model_datetime'] > datetime.fromisoformat('2031-01-01T00:00:00')][['_id', 'model_datetime', 'model_filename']]
# %%


# %%
from bson.objectid import ObjectId
_ids = df[df['model_datetime'] > datetime.fromisoformat('2031-01-01T00:00:00')][['_id']].values
_ids = [['61b80f617d3e96b2d816f266']]
for s in symbols:
    _conf['symbol'] = s
    db_col = _conf['symbol'] + '_' + str(_conf['interval'])
    collection = db[db_col]
    for i in _ids:
        myquery = { "_id": ObjectId(str(i[0])) }
        print(myquery)
        x = collection.find_one(myquery)
        if x is None:
            pass
        else:
            y = collection.delete_one(x)
        print(y.deleted_count, " documents deleted.")

#%%
client.close()
# %%
from pymongo.