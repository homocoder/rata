# %%
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element) -> bool:
    try:
        int(element)
        return True
    except ValueError:
        return False

# %%
from sys import argv

fake_argv = 'update_ohlcv_and_forecast.py  --db_host=localhost --db_port=27017 --db_name=rata --symbol=EURUSD --interval=5 --kind=forex'

fake_argv = fake_argv.split()

#argv = fake_argv ####

_conf = dict()
for i in argv[1:]:
    if '=' in i:
        param = i.split('=')
        _conf[param[0].replace('--', '')] = param[1]

for i in _conf:
    b = _conf[i]
    if   b == 'True':
        _conf[i] = True
    elif b == 'False':
        _conf[i] = False
    elif is_int(b):
        _conf[i] = int(b)
    elif is_float(b):
        _conf[i] = float(b)
_conf

# %%
import pandas as pd
import datetime as dt
from rata.marketon import get_data
from pymongo import MongoClient

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['db_name']]
db_col = 'rates_' + _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

# %%
if not db_col in db.list_collection_names():
    hours_back = 80 * _conf['interval']
else:
    mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1).skip(collection.count_documents({}) - 9)
    df = pd.DataFrame(mydoc)
    df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume']).max('query_tstamp')
    df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
    t1 = df['tstamp'].iloc[-1]
    t2 = dt.datetime.utcnow()
    t3 = t2 - t1
    hours_back = (t3.seconds // 3600) + 1

print('Hours back: ', hours_back)

df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], kind=_conf['kind'], hours=hours_back)
df.index = df.index.to_series().apply(dt.datetime.isoformat)
df.reset_index(inplace=True)
df_dict = df.to_dict(orient='records')

# %%
for r in df_dict:
    collection.update_one(r, {'$set': r}, upsert=True)
    #collection.insert_one(r, {'$set': r})

# %%
mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1).limit(2100)
df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp', ]).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df

# %%
# predict here!

df_dict

# %%



