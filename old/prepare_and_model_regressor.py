# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'prepare_and_model_regressor.py --db_host=localhost --db_port=27017 --db_name=rata --symbol=USDJPY --interval=15 '
fake_argv += '--include_raw_rates=False --include_autocorrs=True --include_all_ta=False '
fake_argv += '--forecast_shift=3 --autocorrelation_lag=8 --autocorrelation_lag_step=2 --n_rows=3000 '

fake_argv = fake_argv.split()
#argv = fake_argv ####

_conf = parse_argv(argv)
print(_conf)
## %%
# Global imports
import pandas as pd
import datetime as dt
import numpy as np

t0 = dt.datetime.now().timestamp()

from pymongo import MongoClient

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_rates']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

# %%
mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 5000) # 

df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp', ]).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df_query = df.copy()
del df

# %%
#df_diff_intervals = pd.DataFrame(df_query.index)
df_diff_intervals = pd.DataFrame(df_query['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
#%%
### Feat eng

## Tech indicators
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

df = df.iloc[-_conf['n_rows'] - 300 :]

print('Count Nan2:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
if len(nancols > 0):
    print(len(nancols), nancols.index)
else:
    print(len(nancols))

df.set_index(['tstamp'], inplace=True, drop=True)
df.drop(['interval', 'symbol'], inplace=True, axis=1)

### AUTOCORRSs
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

## RAW RATES
if _conf['include_raw_rates']:
    for i in ['open', 'high', 'low', 'close', 'volume']:
        df.rename({i: 'x_' + i}, inplace=True, axis=1)

# Momentum: RSI, StochasticOscillator
# Trending: MACD, PSAR
# Volume: MFI
# PPO, MFI, 

## Non-supervised
# IF
# KNN
# DBSCAN

# Left only real x_ features
X_columns = list()
for c in df.columns:
    if 'x_' == c[:2]:
        X_columns.append(c)

# %%
### Outputs X, y
# join Symbol1 and Symbol2 here.

# y_target
df['y_close_shift_' + str(_conf['forecast_shift'])] = df['x_close_roc_' + str(_conf['forecast_shift'])].shift(-_conf['forecast_shift']) # to see the future
y_column = 'y_close_shift_' + str(_conf['forecast_shift'])

# Before deleting rows NaNs, save the X_forecast
X_forecast = df[X_columns].iloc[-1:]

# Delete the forecast rowS and consolidate final X and y
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

# Left only real x_ features
X_columns = list()
for c in df.columns:
    if 'x_' == c[:2]:
        X_columns.append(c)

X_forecast = X_forecast[X_columns]
X = df[X_columns]
y = df[y_column]

print('Count Nan6:', X.isna().sum())
print('Final X_columns', X.columns.sort_values())
X_forecast

# %%
### Outputs train & tests 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_forecast.shape, '\n')
print('Count Nan7:', (df.isna().sum()).sum())
nancols = X_train.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols), nancols.index)

#%%
######                  MODELS    ############
# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, r2_score

seed = int(dt.datetime.now().strftime('%S%f'))
model = XGBRegressor(random_state=seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae =  mean_absolute_error(y_true=y_test, y_pred=y_pred)
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
evs = explained_variance_score(y_true=y_test, y_pred=y_pred)
r2 = r2_score(y_true=y_test, y_pred=y_pred)

y_forecast = model.predict(X_forecast)
print(mae, y_forecast)

feature_importance = list()
for feat, importance in zip(X.columns, model.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
df_feature_importance = pd.DataFrame(feature_importance).sort_values(by='score', ascending=False)
# %%
t1 = dt.datetime.now().timestamp()
total_time = t1 - t0
print('Total time:', total_time)

# Regressor Outputs: id,  total_duration, mae*100, rmse, metrics, df_feature_importance{}, df_delta_minutes{}
# mae, mape, evs, r2, df_feature_importance, df_delta_minutes

_conf['id']         = dt.datetime.now()
_conf['model']      = 'XGBRegressor'
_conf['mae']        = mae
_conf['mape']       = mape
_conf['evs']        = evs
_conf['r2']         = r2
_conf['total_time'] = total_time
_conf['feature_importance'] = df_feature_importance.to_dict(orient='records')
_conf['delta_minutes']      = pd.DataFrame(df_delta_minutes).reset_index().to_dict(orient='records')

client.close()

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_regressors']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

collection.insert_one(_conf, {'$set': _conf})
client.close()

# %%
