# %% 🐭
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'featsel.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3 --nrows=300'
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import custom_resample_close, custom_resample_open, custom_resample_volume, check_time_gaps

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@localhost:5432/rata')

symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD']
symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'NZDUSD']

df_join = pd.DataFrame()
for s in symbols:
    sql =  "select * from feateng "
    sql += "where symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += " order by tstamp desc limit " + str(_conf['nrows'])
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    #df.set_index('tstamp', drop=True, inplace=True)
    X_prefix = s + '_' + str(_conf['interval']) + '_'
    for c in df.columns:
        df.rename({c: X_prefix + c}, axis=1, inplace=True)
    
    
    if len(df_join) == 0:
        df_join = df
    else:
        df_join = pd.merge(df_join, df, on='tstamp', how='inner')
df_join.sort_index


#%%
featsel = ['_roc', '_close', '_SROC_']
features_selected = set()
for i in featsel:
    for j in df.columns:
        if (i in j):
            features_selected.add(j)
features_selected = list(features_selected) + ['tstamp']
#%%
sql = 'with '
for s in symbols:
    sql += s + ' as (select '
    for i in df_columns:
        sql += '"' + i + '" as ' + s + '_' + i + ' ,\n'
    sql = sql[:-2]
    sql += " from feateng where symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += "), \n"
sql = sql[:-3]
sql += '\nselect * from ' + ', '.join(symbols)

first_symbol = symbols[0]
sql += ' where ' + first_symbol
for s in symbols:
    sql += '_tstamp=' + s + '_tstamp and ' + s
sql += '_tstamp=' + first_symbol + '_tstamp'
df = pd.read_sql_query(sql, engine)
#%%
df['tstamp'] = df[first_symbol.lower() + '_tstamp']

df = df[[i for i in df.columns if '_tstamp' not in i]]
df = df[[i for i in df.columns if '_symbol' not in i]]
df = df[[i for i in df.columns if '_interval' not in i]]
df = df.sort_values('tstamp')
check_time_gaps(df, _conf)
df.set_index('tstamp', drop=True, inplace=True)
df.to_csv('../' + str(df.index[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(symbols) + '.' + str(len(df)) + '.csv')
#%%


#%%

import pandas as pd
from sklearn.model_selection import train_test_split

# Read a pandas DataFrame
df = pd.read_csv('https://www.dropbox.com/s/rf8mllry8ohm7hh/2022-05-27T20-57-00.AUDUSD_GBPAUD_AUDCHF_GBPNZD_AUDNZD_EURGBP_NZDUSD.8985.csv?dl=1')
df['tstamp'] = pd.to_datetime(df['tstamp'])

time_horizon = 10

target = 'audusd_close'
X = df
y = df[[target]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42334) # y_train not used in TS

# initialize AutoML instance
from flaml import AutoML

automl = AutoML()

# configure AutoML settings
settings = {
    "time_budget": 15,  # total running time in seconds
    'estimator_list':  ['rf'],
    "metric": "rmse",  # primary metric
    "task": "ts_forecast",  # task type
    "log_file_name": "rata_audusd_flaml.log",  # flaml log file
    "eval_method": "holdout",
    "log_type": "all",
    "label": target,
}

# train the model
automl.fit(dataframe=X_train, **settings, period=time_horizon)

# predictions
y_pred = automl.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mae
#%%


#%%
df.sort_values(by='tstamp', ascending=True, inplace=True)
symbol    = df[['symbol'  ]].iloc[0]['symbol']
interval  = int(df[['interval']].iloc[0]['interval']) # always 1

# %%

if interval != _conf['interval']:
    print('\n##### Resampling: ', _conf['symbol'] + "' and r.interval=" + str(_conf['interval']), ' #####')
    interval = _conf['interval']
    print('To interval:', interval)
    resample_rule = str(_conf['interval']) + 'min'
    ts_open   = df[['tstamp', 'open'  ]].resample(resample_rule, on='tstamp').apply(custom_resample_open)
    ts_high   = df[['tstamp', 'high'  ]].resample(resample_rule, on='tstamp').max()['high']
    ts_low    = df[['tstamp', 'low'   ]].resample(resample_rule, on='tstamp').min()['low']
    ts_close  = df[['tstamp', 'close' ]].resample(resample_rule, on='tstamp').apply(custom_resample_close)
    ts_volume = df[['tstamp', 'volume']].resample(resample_rule, on='tstamp').apply(custom_resample_volume)
    
    ts_open.name   = 'open'
    ts_close.name  = 'close'
    ts_volume.name = 'volume'

    df_resample = pd.concat([ts_open, ts_high, ts_low, ts_close, ts_volume], axis=1).sort_index()
    df_resample['symbol']   = symbol
    df_resample['interval'] = interval
    del df
    df = df_resample.copy()
    df.reset_index(drop=False, inplace=True)
    df.dropna(inplace=True)
else:
    print('\n##### Not resampling: ',  _conf['symbol'] + "' and r.interval=" + str(_conf['interval']), ' #####')

check_time_gaps(df, _conf)
# %% 🐭
# Technical Indicators
import ta
df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

#%%
sql =  "delete from feateng where symbol='" + _conf['symbol'] + "' and interval=" + str(_conf['interval'])
engine.execute(sql)
df.to_sql('feateng', engine, if_exists='append', index=False)