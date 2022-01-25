# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates_check.py --db_host=localhost --interval=1 '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt

# %%
from pymongo import MongoClient

client = MongoClient(_conf['db_host'], 27017)
db = client['rates']

for collection in db.list_collection_names():
    mydoc = db[collection].find({})
    df = pd.DataFrame(mydoc)
    df.sort_values(by='tstamp', ascending=True, inplace=True)
    df_diff_intervals = pd.DataFrame(df['tstamp'])
    df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
    df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
    df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

    df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 2]
    print()
    print(df_delta_minutes)



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

from random import random
from time import sleep
sleep(random() * 1)

df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], exchange=exchange, kind=_conf['kind'], hours=hours_back)
df.index = df.index.to_series().apply(dt.datetime.isoformat)
df.reset_index(inplace=True)
df_dict = df.to_dict(orient='records')

for r in df_dict:
    collection.update_one(r, {'$set': r}, upsert=True)
client.close()

# %%


df_diff_intervals = pd.DataFrame(df_query['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
print(df_delta_minutes)