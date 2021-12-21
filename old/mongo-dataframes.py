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
# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'prepare_and_model_binary_classifier.py --db_host=localhost --db_port=27017 --dbs_prefix=rata_test --symbol=EURUSD --interval=5 '
fake_argv += '--include_raw_rates=True --include_autocorrs=True --include_all_ta=True '
fake_argv += '--forecast_shift=5 --autocorrelation_lag=3 --autocorrelation_lag_step=1 --n_rows=3000 '
fake_argv += '--profit_threshold=0.0008 --test_size=0.9 --store_dataset=True '
fake_argv += '--model_datetime=2021-12-13T17:00:00'
fake_argv = fake_argv.split()
argv = fake_argv ####

_conf = parse_argv(argv)
print(_conf)

## %%
# Global imports
import pandas as pd
import datetime as dt
import numpy as np
import pickle
import gzip

t0 = dt.datetime.now().timestamp()

# %%
from pymongo import MongoClient
client = MongoClient(_conf['db_host'], _conf['db_port'])

db = client[_conf['dbs_prefix'] + '_rates']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 5000) # 

df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp', ]).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df = df[df['tstamp'] <= _conf['model_datetime']]
_conf['model_datetime'] = df.iloc[-1]['tstamp'].to_pydatetime() # adjust model_datetime to match the last timestamp
df_query = df.copy()
del df
client.close()