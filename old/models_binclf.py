# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'models_binclf.py --db_host=localhost --db_port=27017 --dbs_prefix=rata --symbol=BITMEX:BNBUSD --interval=5 '
fake_argv += '--include_raw_rates=True --include_autocorrs=True --include_all_ta=True '
fake_argv += '--forecast_shift=7 --autocorrelation_lag=18 --autocorrelation_lag_step=3 --n_rows=3000 '
fake_argv += '--profit_threshold=0.0089 --test_size=0.9 --store_dataset=False '
fake_argv += '--model_datetime=2021-12-24T04:50:00'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!

_conf = parse_argv(argv)
if ':' in _conf['symbol']:
    _conf['symbol'] = _conf['symbol'].split(':')[1]

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

mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 5000) # 

df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp']).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df = df[df['tstamp'] <= _conf['model_datetime']]
_conf['model_datetime'] = df.iloc[-1]['tstamp'].to_pydatetime() # adjust model_datetime to match the last timestamp
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
X_check = df[X_check_columns].copy()
y_check = df[y_check_columns].copy()

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
print('Data Preparation time: ', dataprep_time)

#%%
# */* MODELS */* #
client = MongoClient(_conf['db_host'], _conf['db_port'])
_conf['id_tstamp']             = dt.datetime.now()
_conf['dataprep_time'] = dataprep_time
from sklearnex import patch_sklearn
patch_sklearn()

# %%
# */*   CLF. BIN. BL. BUY.   */* #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, SelectFromModel

model_name = 'skbinclf_buy'
t0 = dt.datetime.now().timestamp()

X_test = X.copy()
y_test = y.mask(y == 2, 0).copy()
y_check[y_column + '_buy'] = y_test
n_pos_labels = len(y_test[y_test == 1])
n_neg_labels = len(y_test) - n_pos_labels

estimator_clf = Pipeline([
  ('scaler', StandardScaler()),
  ('feature_selection6', SelectFromModel(RandomForestClassifier(criterion='entropy', random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1, class_weight='balanced_subsample', n_estimators=30))),
  ('classification', RandomForestClassifier(criterion='entropy', random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1, class_weight='balanced_subsample', n_estimators=90))
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(dt.datetime.now().strftime('%S%f')))
space = dict()
model = GridSearchCV(estimator_clf, space, n_jobs=-1, cv=cv, refit='precision',
                        scoring=['accuracy', 'precision', 'recall'])

model.fit(X_test, y_test) # Buy only

y_pred =  model.predict(X_test)
y_proba = model.predict_proba(X_test) # TODO: store this for easier backtesting

accuracy  = model.cv_results_['mean_test_accuracy'][0]
precision = model.cv_results_['mean_test_precision'][0]
recall    = model.cv_results_['mean_test_recall'][0]

y_forecast = model.predict(X_forecast)
print(accuracy, precision, recall, y_forecast)

final_estimator = model.best_estimator_._final_estimator
feature_importance = list()
for feat, importance in zip(X_test.columns, final_estimator.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
df_feature_importance = pd.DataFrame(feature_importance).sort_values(by='score', ascending=False)

# Save model to disk
model_filename  = '/home/selknam/var/models/' # TODO: hardcoded
model_filename +=  str(_conf['id_tstamp']).replace(' ', '_') + '_' + model_name + '_' + _conf['symbol'] + '_' + str(_conf['interval'])
model_filename +=  '_' + str(round(precision, 2)) + '_' + str(n_pos_labels) + '.pickle.gz'
fd = gzip.open(model_filename, 'wb')
pickle.dump(model, fd)
fd.close()

_conf['model']      = model_name
_conf['accuracy']   = accuracy
_conf['precision']  = precision
_conf['recall']     = recall
_conf['n_pos_labels']  = n_pos_labels

_conf['y_check'] = y_check.reset_index().iloc[-15:].to_dict(orient='records')
_conf['feature_importance'] = df_feature_importance.to_dict(orient='records')
_conf['delta_minutes']      = pd.DataFrame(df_delta_minutes).reset_index().to_dict(orient='records')
_conf['model_filename']  = model_filename

_conf['fit_time'] = dt.datetime.now().timestamp() - t0

# Change to _models_binclf DB
db = client[_conf['dbs_prefix'] + '_models_skbinclf']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]
collection.insert_one(_conf.copy())

if _conf['store_dataset']:
    pass
df_feature_importance

# %%
# */*   CLF. BIN. BL. SELL.  */* #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, SelectFromModel

model_name = 'skbinclf_sell'
t0 = dt.datetime.now().timestamp()

X_test = X.copy()
y_test = y.mask(y == 1, 0).mask(y == 2, 1).copy()
y_check[y_column + '_sell'] = y_test
n_pos_labels = len(y_test[y_test == 1])
n_neg_labels = len(y_test) - n_pos_labels

estimator_clf = Pipeline([
  ('scaler', StandardScaler()),
  ('feature_selection6', SelectFromModel(RandomForestClassifier(criterion='entropy', random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1, class_weight='balanced_subsample', n_estimators=30))),
  ('classification', RandomForestClassifier(criterion='entropy', random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1, class_weight='balanced_subsample', n_estimators=90))
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(dt.datetime.now().strftime('%S%f')))
space = dict()
model = GridSearchCV(estimator_clf, space, n_jobs=-1, cv=cv, refit='precision',
                        scoring=['accuracy', 'precision', 'recall'])

model.fit(X_test, y_test) # Sell only

y_pred =  model.predict(X_test)
y_proba = model.predict_proba(X_test) # TODO: store this for easier backtesting

accuracy  = model.cv_results_['mean_test_accuracy'][0]
precision = model.cv_results_['mean_test_precision'][0]
recall    = model.cv_results_['mean_test_recall'][0]

y_forecast = model.predict(X_forecast)
print(accuracy, precision, recall, y_forecast)

final_estimator = model.best_estimator_._final_estimator
feature_importance = list()
for feat, importance in zip(X_test.columns, final_estimator.feature_importances_):
    feature_importance.append({'feature_name': feat, 'score': importance})
df_feature_importance = pd.DataFrame(feature_importance).sort_values(by='score', ascending=False)

# Save model to disk
model_filename  = '/home/selknam/var/models/' # TODO: hardcoded
model_filename +=  str(_conf['id_tstamp']).replace(' ', '_') + '_' + model_name + '_' + _conf['symbol'] + '_' + str(_conf['interval'])
model_filename +=  '_' + str(round(precision, 2)) + '_' + str(n_pos_labels) + '.pickle.gz'
fd = gzip.open(model_filename, 'wb')
pickle.dump(model, fd)
fd.close()

_conf['model']      = model_name
_conf['accuracy']   = accuracy
_conf['precision']  = precision
_conf['recall']     = recall
_conf['n_pos_labels']  = n_pos_labels

_conf['y_check'] = y_check.reset_index().iloc[-15:].to_dict(orient='records')
_conf['feature_importance'] = df_feature_importance.to_dict(orient='records')
_conf['delta_minutes']      = pd.DataFrame(df_delta_minutes).reset_index().to_dict(orient='records')
_conf['model_filename']  = model_filename

_conf['fit_time'] = dt.datetime.now().timestamp() - t0

# Change to _models_binclf DB
db = client[_conf['dbs_prefix'] + '_models_skbinclf']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]
collection.insert_one(_conf.copy())

if _conf['store_dataset']:
    pass
df_feature_importance
#%%
client.close()

# %%
# ! Notes

# */*   CLF. BIN. RT. SELL.  */* #
# */*   CLF. BIN. RT. SELL.  */* #

# BL: Baseline. Stratified K-Fold.
# GD: Pipeline+Grid Search+Stratified. CV. [scaler, feat selector, estimator[xgb_scale_pos_weight, xgb_lambda, xgb_reg_gamma, xgb_reg_alfa]
# RT: sample_weights. scale_pos_weight. without metrics. prediction stored on RT on db and metrics calculated afterwards
# %%
