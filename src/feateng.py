# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'feateng.py --db_host=localhost --symbol=BTCUSD --interval=3 '
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
    interval = _conf['interval']
    print('To interval:', interval)
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
print('\n##### Checking time gaps for: ', collection, 'To interval:', interval, ' #####')
df_diff_intervals = pd.DataFrame(df['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 3]
print('Len: ', len(df_delta_minutes))
if len(df_delta_minutes > 0):
    print('First: ', df.iloc[0]['tstamp'])
    print('Last: ',  df.iloc[-1]['tstamp'])
print('Gaps are in minutes. Showing gaps interval*3')
print(df_delta_minutes)
# %% 🐭
# Technical Indicators
# TODO: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
import ta
MACD = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=10, window_slow=21, window_sign=9)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=7, window_slow=18, window_sign=7)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=8, window_slow=21, window_sign=8)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

KST      = ta.trend.KSTIndicator(close=df['close'])
kst_sig  = pd.DataFrame(KST.kst_sig())
kst      = pd.DataFrame(KST.kst())
kst_diff = pd.DataFrame(KST.kst_diff())
df = pd.concat([df, kst, kst_sig, kst_diff], axis=1)

RSI  = ta.momentum.rsi(close=df['close'])
KAMA = ta.momentum.kama(close=df['close'])
OBV  = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])
df = pd.concat([df, RSI, KAMA, OBV], axis=1)
# %%
# Rename columns to X_AUDCHF_5_rsi ... etc...
for c in df.columns:
    df.rename({c: 'X_' + symbol + '_' + str(interval) + '_' + c}, axis=1, inplace=True)
df_feateng = df.copy()
del df
# %%

from unicodedata import normalize
file_name = '../tvdata/BTCUSD_3m_Momentum_Strategy_2022-03-15_b67ff.csv'
df = pd.read_csv(file_name)

columns = list()
for text in df.columns:
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    for c in ['#', '/', '$', '%', ' ', '.', '-']:
        text = text.replace(c, '')
    text = normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8").lower()
    columns.append(text)

df.columns = columns
df = df[['datetime', 'trade', 'signal', 'type', 'price', 'profit']].reset_index(drop=True)

#%%
df1 = df[df['type'].str.contains('Entry')].set_index('trade')
df2 = df[df['type'].str.contains('Exit')].set_index('trade')
df = df1.join(df2, rsuffix='_exit').reset_index()
df_tv = df.copy()
del df
# %%
df_tv['tstamp'] = pd.to_datetime(df_tv['datetime'])
#df_resample = df_tv.resample('1min', on='tstamp').min()['tstamp', 'datetime', 'trade', 'signal', 'type', 'price', 'profit']
#ts_high   = df[['tstamp', 'high'  ]].resample(resample_rule, on='tstamp').max()['high']
# %%
df_feateng.set_index('X_BTCUSD_3_tstamp', inplace=True)
df_tv.set_index('tstamp', inplace=True)
df_feat = df_feateng.join(df_tv).fillna(axis='rows', method='ffill')

# %%
first_tstamp = df_tv.index[0]

# %%
