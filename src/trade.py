# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'trade.py --db_host=localhost --db_port=27017 --dbs_prefix=rata --symbol=ETHUSD --interval=5 --trade_datetime=2021-12-29T00:00:00'
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt
import datetime
from pymongo import MongoClient
from rata.utils import sort_human

# %%
client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_models_skbinclf']

columns = ['symbol', 'interval', 'model_datetime', 'y_check']

df_out= pd.DataFrame()

query = {'model_datetime': {'$gt': _conf['trade_datetime']}}
proyection = {'feature_importance': 0, 'delta_minutes': 0}
for collection in [_conf['symbol'] + '_' + str(_conf['interval'])]:
    mydoc = db[collection].find(query)    
    df = pd.DataFrame(mydoc)[columns]
    df_out = df_out.append(df)

# %%
df_group = df_out.groupby(['symbol', 'interval', 'model_datetime'], as_index=False)
df_group = df_group.max()[['symbol', 'interval', 'model_datetime']]
# %%
df_merge = set()

for symbol, interval, forecast_datetime in df_group.values:
    df = df_out[(df_out['symbol'] == symbol) &
                (df_out['interval'] == interval) &
                (df_out['model_datetime'] == forecast_datetime)]

    for i in df['y_check'].values:
        for j in i:
            row = j
            j['symbol'] = symbol
            j['interval'] = interval
            df_merge.add(str(j))

df_merge = list(df_merge)
df_merge = [eval(i) for i in df_merge]    
df_merge = pd.DataFrame(df_merge)

y_check_columns = list()
for i in df_merge.columns:
    if not ('invest' in i):
        y_check_columns.append(i)
y_check_columns = sort_human(y_check_columns)
df_merge = df_merge[y_check_columns].sort_values(['interval', 'symbol', 'tstamp'])

df_models = df_merge.groupby(['tstamp', 'symbol', 'interval'], as_index=False, sort=True).mean()
# %%
client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_forecasts_skbinclf']

columns = ['symbol', 'interval', 'model_datetime', 'include_raw_rates',
           'include_autocorrs', 'include_all_ta', 'autocorrelation_lag',
           'autocorrelation_lag_step', 'n_rows', 'test_size', 'profit_threshold',
           'forecast_shift', 'model', 'accuracy', 'precision', 'recall',
           'n_pos_labels', 'model_how_old', 'forecast_datetime', 'y_proba_1']

df_out= pd.DataFrame()

filter     = {'model_datetime': {'$gt': _conf['trade_datetime']}}
proyection = {'y_check': 0, 'feature_importance': 0, 'delta_minutes': 0}
for collection in [_conf['symbol'] + '_' + str(_conf['interval'])]:
    mydoc = db[collection].find(filter, proyection)
    df = pd.DataFrame(mydoc)[columns]
    df_out = df_out.append(df)

# %%
df_models['forecast_datetime'] = df_models['tstamp']
df_forecasts = df_out.merge(df_models, on=['symbol', 'interval', 'forecast_datetime']).sort_values(['symbol', 'interval', 'tstamp'])

# %%
# Change to _datasets_binclf DB
db = client[_conf['dbs_prefix'] + '_trade']
db_col = _conf['symbol'] + '_' + str(_conf['interval'])
collection = db[db_col]

collection.insert_many(df_forecasts.to_dict(orient='records'))
# %%
df = df_forecasts[['profit_threshold', 'forecast_shift', 'model', 'precision', 'recall', 'tstamp', 'y_proba_1', 'y_close_shift_6_rolling_sum', 'y_close_shift_12_rolling_sum',
       'n_pos_labels', 'model_how_old', 'forecast_datetime', 'y_close_shift_6', 'y_close_shift_12', 'accuracy']].reset_index(drop=True)
df[(df['precision'] > 0.60) & (df['recall'] > 0.60) & (df['y_proba_1'] > 0.60) & (df['profit_threshold'] > 0.0001)]
