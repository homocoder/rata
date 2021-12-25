# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'prepare_and_model_classifier.py --db_host=localhost --db_port=27017 --dbs_prefix=rata_test --symbol=USDJPY --interval=5 '
fake_argv += '--include_raw_rates=True --include_autocorrs=True --include_all_ta=True '
fake_argv += '--forecast_shift=5 --autocorrelation_lag=18 --autocorrelation_lag_step=5 --n_rows=3000 '
fake_argv += '--profit_threshold=0.001 --test_size=0.5 --store_dataset=True'

fake_argv = fake_argv.split()
argv = fake_argv ####

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

# Tech indicators

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

X_check_columns = ['x_open', 'x_high', 'x_low', 'x_close', 'x_volume']
y_check_columns = list()

# y_target
y_column = 'y_close_shift_' + str(_conf['forecast_shift'])
y_check_columns.append(y_column)
#df[y_column] = df['x_close_roc_' + str(_conf['forecast_shift'])].shift(-_conf['forecast_shift']) # to see the future
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
X_check = df[X_check_columns]
y_check = df[y_check_columns]

print('Count Nan6:', X.isna().sum())
print('Final X_columns', X.columns.sort_values())
X_forecast

# %%
#df_diff_intervals = pd.DataFrame(df_query.index)
df_diff_intervals = pd.DataFrame(df_query['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
df_delta_minutes

# %%
#df_diff_intervals = pd.DataFrame(df_query.index)
df_diff_intervals = pd.DataFrame(X.reset_index()['tstamp'])
df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > int(_conf['interval'])]
df_delta_minutes
# %%
### Outputs train & tests 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_conf['test_size'], shuffle=False)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_forecast.shape, '\n')
print('Count Nan7:', (df.isna().sum()).sum())
nancols = X_train.isna().sum()
nancols = nancols[nancols > 0]
print(len(nancols), nancols.index)

#%%
######                  MODELS    ############
client.close()

client = MongoClient(_conf['db_host'], _conf['db_port'])

# %%
# MultiClass classifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score

seed = int(dt.datetime.now().strftime('%S%f'))

model = XGBClassifier(random_state=seed, validate_parameters=True, booster='gbtree',
         use_label_encoder=False)
#scale_pos_weight
# lamdba
# alfa

model.fit(X_train, y_train)
y_pred =  model.predict(X_test)
y_proba = model.predict_proba(X_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
precision_buy  = precision_score(y_true=y_test, y_pred=y_pred, average='weighted', labels=[1])
recall_buy     = recall_score   (y_true=y_test, y_pred=y_pred, average='weighted', labels=[1])
accuracy_buy   = balanced_accuracy_score (y_true=y_test, y_pred=y_pred)
precision_sell = precision_score(y_true=y_test, y_pred=y_pred, average='weighted', labels=[2])
recall_sell    = recall_score   (y_true=y_test, y_pred=y_pred, average='weighted', labels=[2])
accuracy_sell  = balanced_accuracy_score (y_true=y_test, y_pred=y_pred)

y_forecast = model.predict(X_forecast)
print(precision_buy, recall_buy, accuracy_buy, y_forecast)
print(precision_sell, recall_sell, accuracy_sell, y_forecast)

Xy_test = X_test
Xy_test['y_test'] = y_test
Xy_test['y_pred'] = y_pred
Xy_test['y_proba_0'] = y_proba[ : , 0]
Xy_test['y_proba_1'] = y_proba[ : , 1] 
Xy_test['y_proba_2'] = y_proba[ : , 2]

Xy = Xy_test.join(other=X_check, lsuffix='L', rsuffix='R', how='outer').join(other=y_check, lsuffix='L', rsuffix='R', how='outer')

feature_importance = list()
for feat, importance in zip(X.columns, model.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
df_feature_importance = pd.DataFrame(feature_importance).sort_values(by='score', ascending=False)

# %%

t1 = dt.datetime.now().timestamp()
total_time = t1 - t0
print('Total time:', total_time)

_conf['id']             = dt.datetime.now()
_conf['model']          = 'XGB_MultiClass'
_conf['precision_buy']  = precision_buy 
_conf['recall_buy']     = recall_buy
_conf['accuracy_buy']   = accuracy_buy
_conf['precision_sell'] = precision_sell
_conf['recall_sell']    = recall_sell
_conf['accuracy_sell']  = accuracy_sell

_conf['total_time'] = total_time
_conf['feature_importance'] = df_feature_importance.to_dict(orient='records')
_conf['delta_minutes']      = pd.DataFrame(df_delta_minutes).reset_index().to_dict(orient='records')

# Change to _classifiers DB
db = client[_conf['dbs_prefix'] + '_classifiers']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]
collection.insert_one(_conf, {'$set': _conf})
if [_conf['store_dataset']]:
    # Change to _datasets DB
    db = client[_conf['dbs_prefix'] + '_classifiers_datasets']
    db_col = _conf['symbol'] + '_' + str(_conf['interval'])
    collection = db[db_col]
    Xy['id'] = _conf['id']
    df_dict = Xy.to_dict(orient='records')
    for r in df_dict:
        pass
        collection.update_one(r, {'$set': r}, upsert=True)

# TODO: test multiclass and commit
# %%
# Binary classifier. BUY
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

seed = int(dt.datetime.now().strftime('%S%f'))

model = XGBClassifier(random_state=seed, validate_parameters=True, booster='gbtree',
        objective='binary:logistic', eval_metric=['logloss', 'error'], use_label_encoder=False)
#scale_pos_weight
# lamdba
# alfa

model.fit(X_train, y_train)
y_pred =  model.predict(X_test)
y_proba = model.predict_proba(X_test)

tn, fp, fn, tp  = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
precision = precision_score(y_true=y_test, y_pred=y_pred)
recall    = recall_score(y_true=y_test, y_pred=y_pred)
accuracy  = accuracy_score(y_true=y_test, y_pred=y_pred)

y_forecast = model.predict(X_forecast)
print(precision, recall, y_forecast)

Xy_test = X_test
Xy_test['y_test'] = y_test
Xy_test['y_pred'] = y_pred
Xy_test['y_proba_0'] = y_proba[ : , 0]
Xy_test['y_proba_1'] = y_proba[ : , 1]
Xy_test['y_proba_2'] = y_proba[ : , 2]

Xy = Xy_test.join(other=X_check, lsuffix='L', rsuffix='R', how='outer').join(other=y_check, lsuffix='L', rsuffix='R', how='outer')

feature_importance = list()
for feat, importance in zip(X.columns, model.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
df_feature_importance = pd.DataFrame(feature_importance).sort_values(by='score', ascending=False)

client.close()