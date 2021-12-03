# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'prepare_dataset.py --db_host=localhost --db_port=27017 --db_name=rata --symbol=EURUSD --interval=5'
fake_argv = fake_argv.split()
argv = fake_argv ####

_conf = parse_argv(argv)

## %%
# Global imports
import pandas as pd
import datetime as dt

from pymongo import MongoClient

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['db_name']]
db_col = 'rates_' + _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

# %%
mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 5000) # 

df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp', ]).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df_query = df.copy()
del df

#%%
### Feat eng
import ta

## Tech indicators
#hyperparameters: n_rows, test_size, autocorrelation_lag, autocorrelation_lag_step, include_raw_rates, include_volume
n_rows = 3000
autocorrelation_lag = 10
autocorrelation_lag_step = 1
include_raw_rates = True
window_fast = 12
window_slow = 26
window_sign = 9
forecast_shift = 1
include_autocorrs = True

df = df_query.copy()

# Clean NaN values
df = ta.utils.dropna(df)

# Add ta features
df = ta.add_all_ta_features(
    df, open="open", high="high", low="low", close="close", volume="volume", fillna=False)

print('Count Nan post add all ta features:', (df.isna().sum()).sum())

df.drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)
df = df.iloc[-n_rows:]

print('Count Nan2:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols), nancols.index)

df.set_index(['tstamp'], inplace=True, drop=True)
df.drop(['interval', 'symbol'], inplace=True, axis=1)

### MACDs

#MACD = ta.trend.MACD(close=df['close'], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign)
#column_name     = 'MACD_macd_'   + str(window_fast) + '_'  + str(window_slow) + '_'  + str(window_sign)
#df[column_name] = MACD.macd()
#column_name     = 'MACD_diff_'   + str(window_fast) + '_'  + str(window_slow) + '_'  + str(window_sign)
#df[column_name] = MACD.macd_diff()
#column_name     = 'MACD_signal_' + str(window_fast) + '_'  + str(window_slow) + '_'  + str(window_sign)
#df[column_name] = MACD.macd_signal()

### AUTOCORRSs
if include_autocorrs:
    for c in df.columns:
        for i in range(1, autocorrelation_lag, autocorrelation_lag_step):
            df['x_' + str(c) + '_roc_' + str(i)] = df[c].pct_change(i)
else:   
    for c in ['close']: # close is mandatory
        for i in range(1, autocorrelation_lag, autocorrelation_lag_step):
            df['x_' + str(c) + '_roc_' + str(i)] = df[c].pct_change(i)

# Avoid fragmentation PerformanceWarning
df2 = df.copy()
del df
df = df2.copy()
del df2
print(df.shape)

## RAW RATES
if include_raw_rates:
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

# y_target
df['y_close_shift_' + str(forecast_shift)] = df['x_close_roc_' + str(forecast_shift)].shift(-forecast_shift) # to see the future
y_column = 'y_close_shift_1'

# Before deleting NaNs, save the X_forecast
X_forecast = df[X_columns].iloc[ -1: , : ]

# Delete the forecast row and consolidate final X and y
df.drop(X_forecast.index, inplace=True)

# Delete NaNs and consolidate final X and y: AVOID THIS!
#df = df.iloc[2000:]
df.to_excel('with-nans.xlsx')
print('Count Nan3:', (df.isna().sum()).sum())
nancols = df.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols), nancols.index)
df.dropna(inplace=True) # TODO: BAD IDEA
df.to_excel('without-nans.xlsx')

X = df[X_columns]
y = df[y_column]

print('Count Nan:', X.isna().sum())
print('Final X_columns', X.columns.sort_values())

# %%
### Outputs train & tests 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_forecast.shape, '\n')
X_forecast

#%%
######                  MODELS    ############
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = RandomForestRegressor(n_estimators=100, max_features=0.8, max_samples=0.8, bootstrap=True, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
model.feature_importances_
print(mae, y_forecast)

feature_importance = list()
for feat, importance in zip(X.columns, model.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
pd.DataFrame(feature_importance).sort_values(by='score', ascending=False).head(30)

# %%
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = CatBoostRegressor(loss_function='RMSE', random_state=random_state)
model.fit(X_train, y_train, verbose=0)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
model.feature_importances_
mae, y_forecast

# %%
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = XGBRFRegressor(n_jobs=-1, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
model.feature_importances_
mae, y_forecast

# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = XGBRegressor(n_jobs=-1, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)

mae, y_forecast


# %%
#from rata.marketoff import tpot_conf
#from tpot import TPOTRegressor
#from sklearn.metrics import mean_absolute_error

#random_state = int(dt.datetime.now().strftime('%S%f'))
#model = TPOTRegressor(n_jobs=-1, max_time_mins=3, random_state=random_state, verbosity=1, config_dict=tpot_conf.regressor_config_dict_pca)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
#y_forecast = model.predict(X_forecast)
#mae, y_forecast

# %%

# %%
