# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'feateng.py --db_host=localhost --symbol=BTCUSD --interval=5 '
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
list_collection_names = db.list_collection_names()
list_collection_names.sort()

collection = _conf['symbol'] + '_1'
mydoc = db[collection].find({})
df = pd.DataFrame(mydoc)
df.sort_values(by='tstamp', ascending=True, inplace=True)
symbol    = df[['symbol'  ]].iloc[0]['symbol']
interval  = int(df[['interval']].iloc[0]['interval']) # always 1

# %%
def custom_resample_open(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike.iloc[0]

def custom_resample_close(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike.iloc[-1]

if interval != _conf['interval']:
    print('\n##### Resampling: ', collection, ' #####')
    resample_rule = str(_conf['interval']) + 'min'  
    ts_open   = df[['tstamp', 'open'  ]].resample(resample_rule, on='tstamp').apply(custom_resample_open)['open']
    ts_high   = df[['tstamp', 'high'  ]].resample(resample_rule, on='tstamp').max()['high']
    ts_low    = df[['tstamp', 'low'   ]].resample(resample_rule, on='tstamp').min()['low']
    ts_close  = df[['tstamp', 'close' ]].resample(resample_rule, on='tstamp').apply(custom_resample_close)['close']
    ts_volume = df[['tstamp', 'volume']].resample(resample_rule, on='tstamp').sum()['volume']

    df_resample = pd.concat([ts_open, ts_high, ts_low, ts_close, ts_volume], axis=1).sort_index()
    df_resample['symbol']   = symbol
    df_resample['interval'] = interval
    del df
    df = df_resample.copy()
    df.reset_index(drop=False, inplace=True)
    df.dropna(inplace=True)
else:
    print('\n##### Not resampling: ', collection, ' #####')

# %%
print('\n##### Checking time gaps for: ', collection, ' #####')
df_diff_intervals = pd.DataFrame(df['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 3]
print('Len: ', len(df_delta_minutes))
if len(df_delta_minutes > 0):
    print('First: ', df.iloc[0]['tstamp'])
    print('Last: ',  df.iloc[-1]['tstamp'])
print(df_delta_minutes)
# %%
