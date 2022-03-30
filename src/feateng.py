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

def join_indicators_tv(df_indicators, df_tv):
    # Join df indicators and df tv strategies
    df_indicators.sort_index(inplace=True)
    first_tstamp_indicators = df_indicators.index[0]
    last_tstamp_indicators  = df_indicators.index[-1]

    df_tv.sort_index(inplace=True)
    df_tv.drop('tstamp', axis=1, inplace=True)
    first_tstamp_tv = df_tv.index[0]
    last_tstamp_tv  = df_tv.index[-1]

    df_tv.drop(df_tv.tail(1).index, inplace=True) # drop last 2 rows
    df_tv = df_tv[(df_tv.index >= first_tstamp_indicators) & (df_tv.index <= last_tstamp_indicators)]
    df_indicators = df_indicators[(df_indicators.index >= first_tstamp_tv) & (df_indicators.index <= last_tstamp_tv)]

    df_feateng = df_indicators.join(df_tv).fillna(axis='rows', method='ffill').sort_index()
    df_feateng.dropna(inplace=True)
    df_feateng['tstamp'] = df_feateng.index
    return df_feateng

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

def custom_resample_volume(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike.drop_duplicates()

def check_time_gaps(df):
    print('\n##### Checking time gaps for: ', collection, 'To interval:', interval, ' #####')
    df_diff_intervals = pd.DataFrame(df['tstamp'])
    df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
    df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
    df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

    df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 1]
    print('Len: ', len(df_delta_minutes))
    if len(df_delta_minutes > 0):
        print('First: ', df.iloc[0]['tstamp'])
        print('Last: ',  df.iloc[-1]['tstamp'])
    print('Gaps are in minutes. Showing gaps interval*3')
    print(df_delta_minutes)

def read_tv_strategy(file_name):
    from unicodedata import normalize
    from pandas import read_csv
    
    df = read_csv(file_name)

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

    strategy_name = file_name.split('_')[-4]

    df.columns = columns
    df = df[['datetime', 'trade', 'signal', 'type', 'price', 'profit']].reset_index(drop=True)
    df1 = df[df['type'].str.contains('Entry')].set_index('trade')
    df2 = df[df['type'].str.contains('Exit')].set_index('trade')
    df = df1.join(df2, rsuffix='_exit').reset_index()
    datetime_tmp = pd.to_datetime(df['datetime']).dt.round(freq='3min')

    for c in df.columns:
        df.rename({c: c + '_' + strategy_name}, axis=1, inplace=True)
    df['tstamp'] = datetime_tmp

    df.set_index(df['tstamp'], inplace=True)
    return df

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

if interval != _conf['interval']:
    print('\n##### Resampling: ', collection, ' #####')
    interval = _conf['interval']
    print('To interval:', interval)
    resample_rule = str(_conf['interval']) + 'min'
    ts_open   = df[['tstamp', 'open'  ]].resample(resample_rule, on='tstamp').apply(custom_resample_open)['open']
    ts_high   = df[['tstamp', 'high'  ]].resample(resample_rule, on='tstamp').max()['high']
    ts_low    = df[['tstamp', 'low'   ]].resample(resample_rule, on='tstamp').min()['low']
    ts_close  = df[['tstamp', 'close' ]].resample(resample_rule, on='tstamp').apply(custom_resample_close)['close']
    ts_volume = df[['tstamp', 'volume']].resample(resample_rule, on='tstamp').apply(custom_resample_volume)['volume']
    ts_volume = ts_volume.apply(lambda x : x.sum())

    df_resample = pd.concat([ts_open, ts_high, ts_low, ts_close, ts_volume], axis=1).sort_index()
    df_resample['symbol']   = symbol
    df_resample['interval'] = interval
    del df
    df = df_resample.copy()
    df.reset_index(drop=False, inplace=True)
    df.dropna(inplace=True)
else:
    print('\n##### Not resampling: ', collection, ' #####')

df = df[-2500:]
check_time_gaps(df)
# %% 🐭
# Technical Indicators
# TODO: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
import ta
for c in df.columns.drop(['tstamp', 'symbol', 'interval']):
    for i in range(3, 10, 3):
        df[str(c) + '_ROC_' + str(i)] = df[c].pct_change(i) * 100

df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
df.set_index(df['tstamp'], inplace=True)
df_feateng = df.copy()
del df
# %%
X_prefix = 'X_' + symbol + '_' + str(interval) + '_'
for c in df_feateng.columns:
    df_feateng.rename({c: X_prefix + c}, axis=1, inplace=True)

Y_prefix = 'Y_' + symbol + '_' + str(interval) + '_'
# %%

# Y for regression
df_feateng[Y_prefix + 'close_ROC_3_shift_3'] = df_feateng[X_prefix + 'close_ROC_3'].shift(-3) # to see the future
df_feateng[Y_prefix + 'close_ROC_6_shift_6'] = df_feateng[X_prefix + 'close_ROC_6'].shift(-6) # to see the future
df_feateng[Y_prefix + 'close_ROC_9_shift_9'] = df_feateng[X_prefix + 'close_ROC_9'].shift(-9) # to see the future

# Y for classification
df_feateng[Y_prefix + 'close_ROC_3_shift_3_B'] = 0
df_feateng[Y_prefix + 'close_ROC_3_shift_3_B'] = df_feateng[Y_prefix + 'close_ROC_3_shift_3_B'].mask(df_feateng[Y_prefix + 'close_ROC_3_shift_3'] > 0.3, 1)

df_feateng[Y_prefix + 'close_ROC_6_shift_6_B'] = 0
df_feateng[Y_prefix + 'close_ROC_6_shift_6_B'] = df_feateng[Y_prefix + 'close_ROC_6_shift_6_B'].mask(df_feateng[Y_prefix + 'close_ROC_6_shift_6'] > 0.3, 1)

df_feateng[Y_prefix + 'close_ROC_9_shift_9_B'] = 0
df_feateng[Y_prefix + 'close_ROC_9_shift_9_B'] = df_feateng[Y_prefix + 'close_ROC_9_shift_9_B'].mask(df_feateng[Y_prefix + 'close_ROC_9_shift_9'] > 0.3, 1)


df_feateng[Y_prefix + 'close_ROC_3_shift_3_S'] = 0
df_feateng[Y_prefix + 'close_ROC_3_shift_3_S'] = df_feateng[Y_prefix + 'close_ROC_3_shift_3_S'].mask(df_feateng[Y_prefix + 'close_ROC_3_shift_3'] < -0.3, 1)

df_feateng[Y_prefix + 'close_ROC_6_shift_6_S'] = 0
df_feateng[Y_prefix + 'close_ROC_6_shift_6_S'] = df_feateng[Y_prefix + 'close_ROC_6_shift_6_S'].mask(df_feateng[Y_prefix + 'close_ROC_6_shift_6'] < -0.3, 1)

df_feateng[Y_prefix + 'close_ROC_9_shift_9_S'] = 0
df_feateng[Y_prefix + 'close_ROC_9_shift_9_S'] = df_feateng[Y_prefix + 'close_ROC_9_shift_9_S'].mask(df_feateng[Y_prefix + 'close_ROC_9_shift_9'] < -0.3, 1)

X_pred = df_feateng[-1:]
df_feateng.dropna(inplace=True)
df_feateng['tstamp'] = df_feateng.index
check_time_gaps(df_feateng)

# END OF FEATENG

# %%
# START OF FEATSEL
featsel = ['kst_diff',
'momentum_pvo',#
'momentum_pvo_hist',#
'momentum_pvo_signal',#
'momentum_tsi',
'momentum_uo',
'trend_adx',
'trend_adx_neg',
'trend_adx_pos',
'trend_aroon_ind',
'trend_dpo',
'trend_ichimoku_b',
'trend_kst_sig',
'trend_mass_index',
'trend_psar_down',
'trend_psar_up',
'trend_sma_slow',
'trend_stc',
'trend_visual_ichimoku_a',
'trend_visual_ichimoku_b',
'volatility_atr',
'volatility_bbw',
'volatility_dcw',
'volatility_kcw',
'volatility_ui',
'volume_adi',
'volume_cmf',
'volume_nvi',
'volume_obv',
'volume_sma_em',
'_close' #
]
features_selected = set()
for i in featsel:
    for j in df_feateng.columns:
        if (i in j) and ('Y_' not in j):
            features_selected.add(j)
features_selected = list(features_selected)

X_columns  = features_selected
y_column_close_ROC_3_shift_3_B = Y_prefix + 'close_ROC_3_shift_3_B'
y_column_close_ROC_3_shift_3_S = Y_prefix + 'close_ROC_3_shift_3_S'

y_column_close_ROC_6_shift_6_B = Y_prefix + 'close_ROC_6_shift_6_B'
y_column_close_ROC_6_shift_6_S = Y_prefix + 'close_ROC_6_shift_6_S'

y_column_close_ROC_9_shift_9_B = Y_prefix + 'close_ROC_9_shift_9_B'
y_column_close_ROC_9_shift_9_S = Y_prefix + 'close_ROC_9_shift_9_S'

# %%
# START MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score

y_columns = [y_column_close_ROC_3_shift_3_B,  y_column_close_ROC_3_shift_3_S,
             y_column_close_ROC_6_shift_6_B,  y_column_close_ROC_6_shift_6_S,
             y_column_close_ROC_9_shift_9_B, y_column_close_ROC_9_shift_9_S]

X      = df_feateng[X_columns].reindex(sorted(df_feateng[X_columns].columns), axis=1)
X_pred = X_pred[X_columns].reindex(sorted(X_pred[X_columns].columns), axis=1)

# %%
# FORECAST 2500, 10
df_forecast = X_pred
for y_column in y_columns:
    y = df_feateng[y_column]
    # create model
    clf = RandomForestClassifier(random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1)
    clf.class_weight = "balanced"
    clf.fit(X, y)
    y_proba = clf.predict_proba(X_pred)
    y_proba = pd.DataFrame(y_proba, columns=[y_column + '_proba_0', y_column + '_proba_1'])
    y_proba.set_index(X_pred.index, inplace=True)
    df_forecast = df_forecast.join(y_proba)
df_forecast['tstamp_forecast'] = dt.datetime.now()
df_forecast.to_csv('BTCUSD_3.forecast_1.csv', mode='a', header=False)
# %%
