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

# %%
from sys import argv

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
#hyperparameters: test_size


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
# LSTM
