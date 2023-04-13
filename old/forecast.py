# %% 🐭
from sys import argv
from rata.utils import parse_argv, split_sequences

fake_argv  = 'forecast.py --db_host=192.168.1.84 --symbol=AUDUSD --interval=3 --nrows=3000 '
fake_argv += '--symbols=AUDUSD,AUDCHF,NZDUSD '
fake_argv += '--X_columns=tstamp,AUDCHF_3_close_SROC_15,NZDUSD_3_close_SROC_15,AUDUSD_3_close_SROC_15,AUDUSD_3_trend_macd_diff '
fake_argv += '--y_column=y_AUDUSD_3_close_SROC_15_shift-15 '
fake_argv += '--n_steps_in=60 --n_steps_out=3 --epochs=10 --test_lenght=1000 '
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)

_conf['X_columns'] = _conf['X_columns'].split(',')
_conf['symbols']   = _conf['symbols'].split(',')

X_columns   = _conf['X_columns']
y_column    = _conf['y_column']
n_steps_in  = _conf['n_steps_in']
n_steps_out = _conf['n_steps_out']
epochs      = _conf['epochs']
test_lenght = _conf['test_lenght']
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import check_time_gaps
import numpy  as np
from sqlalchemy import create_engine
#import tensorflow as tf
import datetime
from sklearn.metrics import mean_squared_error

#%%
engine = create_engine('postgresql+psycopg2://rata:<passwd>@' + _conf['db_host'] + ':5432/rata')

df_join = pd.DataFrame()
for s in _conf['symbols']:
    sql =  "select * from feateng "
    sql += "where symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += " order by tstamp desc limit " + str(_conf['nrows'])
    print(sql)
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    X_prefix = s + '_' + str(_conf['interval']) + '_'
    for c in df.columns:
        df.rename({c: X_prefix + c}, axis=1, inplace=True)
    
    if len(df_join) == 0:
        df_join = df
    else:
        df_join = pd.merge(df_join, df, how='inner', left_on=df_join.columns[0], right_on=df.columns[0])

df = df_join.copy()
df.sort_values(df.columns[0])
df['tstamp'] = df.iloc[:,0]
df.sort_values('tstamp', ascending=True)
check_time_gaps(df, {'symbol': s, 'interval': 3})
df.set_index('tstamp', drop=True, inplace=True)
df.reset_index(drop=False, inplace=True)

print(len(df.iloc[:,0].drop_duplicates()) == len(df.iloc[:,0]))
df.sort_values('tstamp', inplace=True)
df.reset_index(drop=True, inplace=True)
df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

#%%
df[y_column] = df['AUDUSD_3_close_SROC_15'].shift(-16) # -15 (-1) # TODO: hardcoded

# X forecast
tmplist = list()
tmplist.append(df[X_columns][-n_steps_in:].drop('tstamp', axis=1).values)
X_forecast = np.array(tmplist)
tstamp_forecast = df['tstamp'].max()

# X, y train
Xy_train             = df[:-test_lenght][X_columns]
Xy_train[y_column]   = df[:-test_lenght][y_column]
dataset_train        = Xy_train.drop('tstamp', axis=1).values
X_train, y_train     = split_sequences(dataset_train, n_steps_in, n_steps_out)

# X, y test
Xy_test              = df[-test_lenght:-16][X_columns] # TODO: hardcoded
Xy_test[y_column]    = df[-test_lenght:-16][y_column]  # TODO: hardcoded
dataset_test         = Xy_test.drop('tstamp', axis=1).values
X_test, y_test       = split_sequences(dataset_test, n_steps_in, n_steps_out)
dataset_test_ts      = Xy_test.values
X_test_ts, y_test_ts = split_sequences(dataset_test_ts, n_steps_in, n_steps_out)

#%%
# Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# the dataset knows the number of features, e.g. 2
n_features = X_train.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
t1 = datetime.datetime.now()
#model.fit(X_train, y_train, epochs=epochs, verbose=1, callbacks=[tensorboard_callback], use_multiprocessing=True)
model.fit(X_train, y_train, epochs=epochs, verbose=1, use_multiprocessing=True)
t = int((datetime.datetime.now() - t1).total_seconds() / 60)

#%%
# Last sequence forecast
y_forecast = model.predict(X_test, verbose=1)
print(X_test_ts.shape, X_test.shape, y_forecast.shape, y_test.shape)
yhat = model.predict(X_forecast, verbose=1)
print(yhat.shape)
yhat
#%%
# DF VALIDATION
df_val = list()
for i in range(len(X_test_ts)):
    for j in range(len(y_forecast[i])):
        df_val.append([X_test_ts[i, -1, 0], j+1, y_test[i, j], y_forecast[i, j]])

df_val = pd.DataFrame(df_val, columns=['tstamp', 'step', 'y_test', 'y_forecast'])
mse = mean_squared_error(df_val['y_test'], df_val['y_forecast'])
print('Global MSE: ', mse)

df_val['symbol']       = _conf['symbol']
df_val['interval']     = _conf['interval']
df_val['nrows']        = _conf['nrows']
df_val['X_columns']    = ','.join(X_columns)
df_val['symbols']      = ','.join(_conf['symbols'])
df_val['y_column']     = y_column
df_val['n_steps_in']   = n_steps_in
df_val['n_steps_out']  = n_steps_out
df_val['epochs']       = epochs
df_val['test_lenght']  = test_lenght
df_val['model_tstamp'] = tstamp_forecast
df_val['model_id']     = str(tstamp_forecast).replace(' ', 'T')
df_val['fit_time']     = t
df_val['mse']          = mse

df_val.reset_index(drop=True)
df_val['forecast_index'] = (df_val.index +  3) // 3
df_val

# MSE per step
from sklearn.metrics import mean_squared_error
steps = max(df_val['step'])
for s in range(1, steps + 1):
    df_mse = df_val[df_val['step'] == s]
    mse = mean_squared_error(df_mse['y_test'], df_mse['y_forecast'])
    print(s, mse)

# MAE per step
from sklearn.metrics import median_absolute_error
steps = max(df_val['step'])
for s in range(1, steps + 1):
    df_mae = df_val[df_val['step'] == s]
    mae = median_absolute_error(df_mae['y_test'], df_mae['y_forecast'])
    print(s, mae)

#%%
df_val[(df_val['y_forecast'] >  0.15) & (df_val['step'] == 1)].reset_index(drop=True)

#%%
df_val[(df_val['y_forecast'] < -0.15) & (df_val['step'] == 1)].reset_index(drop=True)

#%%
engine = create_engine('postgresql+psycopg2://rata:<passwd>@' + _conf['db_host'] + ':5432/rata')

df_val.to_sql('forecast', engine, if_exists='append', index=False)

# %%
df_val
# %%
