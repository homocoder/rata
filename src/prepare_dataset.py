# %%
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element) -> bool:
    try:
        int(element)
        return True
    except ValueError:
        return False

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    from numpy import array
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# %%
from sys import argv
import flaml

fake_argv = 'prepare_dataset.py --db_host=localhost --db_port=27017 --db_name=rata --symbol=EURUSD --interval=5'

fake_argv = fake_argv.split()

argv = fake_argv ####

_conf = dict()
for i in argv[1:]:
    if '=' in i:
        param = i.split('=')
        _conf[param[0].replace('--', '')] = param[1]

for i in _conf:
    b = _conf[i]
    if   b == 'True':
        _conf[i] = True
    elif b == 'False':
        _conf[i] = False
    elif is_int(b):
        _conf[i] = int(b)
    elif is_float(b):
        _conf[i] = float(b)
_conf

# %%
import pandas as pd
import datetime as dt
from rata.marketon import get_data
from pymongo import MongoClient
from tpot import TPOTRegressor # to avoid the f* warning about NN
import tensorflow as tf

from keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.model_selection import train_test_split
from keras.metrics import MeanAbsoluteError, MeanSquaredLogarithmicError

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['db_name']]
db_col = 'rates_' + _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

# %%
mydoc = collection.find({'symbol': _conf['symbol']}).sort('tstamp', 1)#.skip(collection.count_documents({}) - 1009)

df = pd.DataFrame(mydoc)
df = df.groupby(['interval',  'status', 'symbol', 'tstamp', 'unix_tstamp', ]).mean()
df = df.reset_index()[['tstamp', 'interval', 'symbol', 'open', 'high', 'low', 'close', 'volume']].sort_values('tstamp')
df

# %%
autocorrelation_lag = 10
for i in range(1, autocorrelation_lag):
    df['close_roc_' + str(i)] = df['close'].pct_change(i)
df['close_shift_1'] = df['close_roc_1'].shift(-1) # to see the future
df
#dataset = df.dropna().copy()
#dataset
# %%
X_columns = list()
for c in df.columns:
    if 'close_roc' in c:
        X_columns.append(c)

y_column = 'close_shift_1'

X_forecast = df[X_columns].iloc[ -1: , : ]
df.dropna(inplace=True)

X = df[X_columns]
y = df[y_column]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_forecast.shape, '\n')
X_forecast

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = RandomForestRegressor(n_estimators=100, max_features=0.8, max_samples=0.8, bootstrap=True, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast

# %%
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = CatBoostRegressor(loss_function='RMSE', random_state=random_state)
model.fit(X_train, y_train, verbose=0)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast

# %%
#hyperparameters: test_size, autocorrelation_lag


# %%
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = XGBRFRegressor(n_jobs=-1, random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
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
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = TPOTRegressor(n_jobs=-1, max_time_mins=3, random_state=random_state, verbosity=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast

# %%
from rata.marketoff import tpot_conf
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = TPOTRegressor(n_jobs=-1, max_time_mins=3, random_state=random_state, verbosity=1, config_dict=tpot_conf.regressor_config_dict_pca)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast

# %%
from rata.marketoff import tpot_conf
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = TPOTRegressor(n_jobs=-1, max_time_mins=3, random_state=random_state, verbosity=1, config_dict=tpot_conf.regressor_config_dict_xgb)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast

# %%
from rata.marketoff import tpot_conf
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error

random_state = int(dt.datetime.now().strftime('%S%f'))
model = TPOTRegressor(n_jobs=-1, max_time_mins=3, random_state=random_state, verbosity=1, config_dict=tpot_conf.regressor_config_dict_rf)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
y_forecast = model.predict(X_forecast)
mae, y_forecast
# %%
# AutoKerasReggresor
# %%
# AutoKeras Timeseries forecaster

# %%
#FLAML

# %%
# LSTM # Input: X, y
# Config
import numpy as np
nsamples = len(X)
ncolumns = len(X.columns)
# choose a number of time steps
n_steps_in, n_steps_out = 90, 1

# multivariate multi-step data preparation

# define input sequence # convert to [rows, columns] structure

in_seq = list()
for i in range(ncolumns):
    in_seq.append(X.iloc[:, i].values.reshape((nsamples, 1))) ### X here

out_seq = y    ### y here
out_seq = out_seq.values.reshape((len(out_seq), 1))

# horizontally stack columns
in_seq.append(out_seq)
dataset = np.hstack(tuple(in_seq)) # here, in_seq contains in_seq and out_seq, just for var economy

# covert into input/output
XX, YY = split_sequences(dataset, n_steps_in, n_steps_out)
print(XX.shape, YY.shape)

# the dataset knows the number of features, e.g. 2
n_features = XX.shape[2]

XX_train, XX_test, YY_train, YY_test = train_test_split(XX, YY, test_size=0.05, shuffle=False)
print(XX_train.shape, YY_train.shape, XX_test.shape, YY_test.shape)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error

# define model
model = Sequential()
model.add(LSTM(90, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(30, activation='relu', return_sequences=True))
model.add(LSTM(90, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredLogarithmicError()])

# fit model
model.fit(XX, YY, epochs=10, batch_size=128, verbose=1, validation_split=0.05, validation_data=(XX_test, YY_test), shuffle=False, use_multiprocessing=True)
YY_pred = model.predict(XX_test)
mae = mean_absolute_error(y_true=YY_test, y_pred=YY_pred)
#YY_forecast = model.predict(XX_forecast)
mae
# %%
#from flaml import AutoML

#automl = AutoML()
#automl_settings = {
#    "time_budget": 60,
#    "metric": 'mse',
#    "task": 'regression',
#    "log_file_name": 'mylog.log'
#}
#automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
#y_pred = model.predict(X_test)
#mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
#y_forecast = model.predict(X_forecast)
#mae, y_forecast
# %%

# 
# %%
# FLAML task='ts_forecast'
#import numpy as np
#from flaml import AutoML
#X_train = np.arange('2014-01', '2021-01', dtype='datetime64[M]')
#y_train = np.random.random(size=72)
#automl = AutoML()
#automl.fit(X_train=X_train[:72].copy(),  # a single column of timestamp
#           y_train=y_train.copy(),  # value for each timestamp
#           period=12,  # time horizon to forecast, e.g., 12 months
#           task='ts_forecast', time_budget=15,  # time budget in seconds
#           log_file_name="ts_forecast.log",
#          )
#print(automl.predict(X_train[72:]))
# %%
# Ludwig by Uber
# Prophet