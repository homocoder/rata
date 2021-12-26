# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'forecasts_binclf.py --db_host=localhost --db_port=27017 --dbs_prefix=rata_test --symbol=BTCUSD --interval=5 '
fake_argv += '--include_raw_rates=True --include_autocorrs=True --include_all_ta=True '
fake_argv += '--forecast_shift=5 --autocorrelation_lag=18 --autocorrelation_lag_step=3 --n_rows=3000 '
fake_argv += '--profit_threshold=0.0008 --test_size=0.9 --store_dataset=False '
fake_argv += '--forecast_datetime=2031-12-17T19:35:00 '
fake_argv = fake_argv.split()
#argv = fake_argv #### *!

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

mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 2000) # 

df = pd.DataFrame(mydoc)
df = df.groupby(['interval', 'status', 'symbol', 'tstamp', 'unix_tstamp']).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df = df[df['tstamp'] <= _conf['forecast_datetime']]
_conf['forecast_datetime'] = df.iloc[-1]['tstamp'].to_pydatetime() # adjust forecast_datetime to match the last timestamp
df_query = df.copy()
del df
client.close()

#%%
# */* LIB TA */* #
df = df_query.copy()

if _conf['include_all_ta']: 
    import ta

    # Clean NaN values
    df = ta.utils.dropna(df)

    # Add ta features
    df = ta.add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=False)

    df.drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)

print('Count Nan post add all ta features:', (df.isna().sum()).sum())

df = df.iloc[-_conf['n_rows'] - 300 :] # TODO: improve the deletion of first 300 rows

print('Count Nan2:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
if len(nancols > 0):
    print(len(nancols), nancols.index)
else:
    print(len(nancols))

df.set_index(['tstamp'], inplace=True, drop=True)
df.drop(['interval', 'symbol'], inplace=True, axis=1)

# Momentum: RSI, StochasticOscillator
# Trending: MACD, PSAR
# Volume: MFI
# PPO, MFI, Klinger, Fibonacci

## Non-supervised
# IForest
# KNN
# DBSCAN

# */* AUTOCORRS */* #
if _conf['include_autocorrs']:
    for c in df.columns:
        for i in range(1, _conf['autocorrelation_lag'], _conf['autocorrelation_lag_step']):
            df['x_' + str(c) + '_roc_' + str(i)] = df[c].pct_change(i)
else:   
    for c in ['close']: # close is mandatory
        for i in range(1, _conf['autocorrelation_lag'], _conf['autocorrelation_lag_step']):
            df['x_' + str(c) + '_roc_' + str(i)] = df[c].pct_change(i)

# Avoid fragmentation PerformanceWarning
df2 = df.copy()
del df
df = df2.copy()
del df2
print(df.shape)

# */* RAW RATES */* #
if _conf['include_raw_rates']:
    for i in ['open', 'high', 'low', 'close', 'volume']:
        #df.rename({i: 'x_' + i}, inplace=True, axis=1)
        df['x_' + i] = df[i]

# %%
# join Symbol1 and Symbol2 here.

X_check_columns = ['open', 'high', 'low', 'close', 'volume']
y_check_columns = list()

# y_target
y_column = 'y_close_shift_' + str(_conf['forecast_shift'])
y_check_columns.append(y_column)

df[y_column] = df['x_close_roc_1'].shift(-_conf['forecast_shift']) # to see the future
X_check_columns.append('x_close_roc_1')
df[y_column + '_sign'] = df[y_column].mask(df[y_column] > 0, 1).mask(df[y_column] < 0, -1)
df[y_column + '_sign_rolling_sum'] = df[y_column + '_sign'].rolling(_conf['forecast_shift']).sum()
df[y_column + '_rolling_sum'] = df[y_column].rolling(_conf['forecast_shift']).sum()
y_check_columns.append(y_column + '_rolling_sum')

df[y_column + '_invest'] = df[y_column + '_rolling_sum']
df[y_column + '_invest'] = 0
df[y_column + '_invest'] = df[y_column + '_invest'].mask(df[y_column + '_rolling_sum'] >  _conf['profit_threshold'], 1)
df[y_column + '_invest'] = df[y_column + '_invest'].mask(df[y_column + '_rolling_sum'] < -_conf['profit_threshold'], 2)
y_column = y_column + '_invest'
y_check_columns.append(y_column)

#%%
# Keep only real x_ features
X_columns = list()
for c in df.columns:
    if 'x_' == c[:2]:
        X_columns.append(c)

# Before deleting rows NaNs, save the X_forecast
X_forecast = df[X_columns].iloc[-1:]

# Delete the forecast rowS
X_forecast_indexes = df[X_columns].iloc[ -_conf['forecast_shift']: , : ].index
df.drop(X_forecast_indexes, inplace=True)

# Delete NaNs and consolidate final X and y: AVOID THIS!
df = df.iloc[-_conf['n_rows'] : ]
print('Count Nan3:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
print('Count Nan4:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols))

df.dropna(inplace=True, axis=1)  # Delete columns that already have NaNs!!!!!!!
print('Count Nan5:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols))

# Keep only real x_ features. (repeat this step after cleaning NaNs)
X_columns = list()
for c in df.columns:
    if 'x_' == c[:2]:
        X_columns.append(c)

X_forecast = X_forecast[X_columns]
X = df[X_columns]
y = df[y_column]
X_check = df[X_check_columns]
y_check = df[y_check_columns]

print('Count Nan6:', X.isna().sum())
print('Final X_columns', X.columns.sort_values())
X_forecast

# Outputs: X, y, X_check, y_check, X_forecast
# %%

df_diff_intervals = pd.DataFrame(df_query['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
print(df_delta_minutes)

df_diff_intervals = pd.DataFrame(X.reset_index()['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
print(df_delta_minutes)

dataprep_time = dt.datetime.now().timestamp() - t0
print('Data preparation time: ', dataprep_time)

#%%
# */* MODELS */* #
client = MongoClient(_conf['db_host'], _conf['db_port'])

db = client[_conf['dbs_prefix'] + '_models_binclf']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]
mydoc = collection.find({'symbol': _conf['symbol']}).sort('model_datetime', 1)#.skip(collection.count_documents({}) - 12) #
df = pd.DataFrame(mydoc)
print(_conf)
#df.drop(['feature_importance', 'delta_minutes'], axis=1, inplace=True)
df = df[df['model_datetime'] <= _conf['forecast_datetime']]
df['model_how_old'] = (_conf['forecast_datetime'] - df['model_datetime']).dt.total_seconds()
df = df[df['model_how_old'] < 3600] # TODO: hardcoded, 2 hours

# %%
# */*   CLF. BIN. FORECAST.   */* #
model_name = 'xgb_bin_BL_buy'

seed = int(dt.datetime.now().strftime('%S%f'))

if len(df) == 0:
    raise BaseException('No models available for this forecast_datetime')

if X_forecast.index[0].to_pydatetime() != _conf['forecast_datetime']:
    raise BaseException('Requested forecast_datetime is different than the one on dataset')

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_forecasts_binclf']

_conf['id_tstamp']             = dt.datetime.now()

for i in range(0, len(df)):
    t0 = dt.datetime.now().timestamp()
    row = df.iloc[i]
    row.drop('_id', inplace=True)
    model_filename = row['model_filename']
    feature_importance = row['feature_importance']
    X_columns = pd.DataFrame(feature_importance)['feature_name'].values
    
    fd = gzip.open(model_filename, 'rb')
    model = pickle.load(fd)
    fd.close()

    y_forecast = model.predict(X_forecast[X_columns])
    y_proba = model.predict_proba(X_forecast[X_columns])
    print(y_forecast, y_proba)
    row['forecast_datetime'] = _conf['forecast_datetime']
    row['y_proba_0']  = y_proba[ : , 0][0]
    row['y_proba_1']  = y_proba[ : , 1][0]
    row['y_forecast'] = y_forecast[0]
    row['id_tstamp_model'] = row['id_tstamp']
    row['id_tstamp']   = _conf['id_tstamp']
    row['dataprep_time'] = dataprep_time
    row['forecast_time']   = dt.datetime.now().timestamp() - t0
    
    # Change to _forecasts DB
    db_col = _conf['symbol'] + '_' + str(_conf['interval'])
    collection = db[db_col]
    _id = collection.insert_one(row.to_dict())
    _id = str(_id.inserted_id)

#%%
client.close()

# %%
# ! Notes

# */*   CLF. BIN. RT. SELL.  */* #
# */*   CLF. BIN. RT. SELL.  */* #

# BL: Baseline. Stratified K-Fold.
# GD: Pipeline+Grid Search+Stratified. CV. [scaler, feat selector, estimator[xgb_scale_pos_weight, xgb_lambda, xgb_reg_gamma, xgb_reg_alfa]
# RT: sample_weights. scale_pos_weight. without metrics. prediction stored on RT on db and metrics calculated afterwards

#%%
#import pickle

#loaded_model = pickle.load(open('test.model.pickle', 'rb'))