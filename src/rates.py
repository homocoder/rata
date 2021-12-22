# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates.py --db_host=localhost --db_port=27017 --dbs_prefix=rata_test --symbol=BINANCE:BTCEUR --interval=5 --kind=crypto'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
if _conf['kind'] == 'forex':
    exchange = 'OANDA'
if _conf['kind'] == 'crypto':
    exchange = 'COINBASE'
if ':' in _conf['symbol']:
    exchange        = _conf['symbol'].split(':')[0]
    _conf['symbol'] = _conf['symbol'].split(':')[1]
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt

# %%
from pymongo import MongoClient
from rata.marketon import get_data

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_rates']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

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

from random import random
from time import sleep
sleep(random() * 3)

df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], exchange=exchange, kind=_conf['kind'], hours=hours_back)
df.index = df.index.to_series().apply(dt.datetime.isoformat)
df.reset_index(inplace=True)
df_dict = df.to_dict(orient='records')

for r in df_dict:
    collection.update_one(r, {'$set': r}, upsert=True)

# %%
client.close()