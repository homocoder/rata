# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates.py --db_host=localhost --db_port=27017 --dbs_prefix=rt '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt
from pymongo import MongoClient

# %%
client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_forecasts_binclf']

df_out= pd.DataFrame()

for collection in db.list_collection_names():
    mydoc = db[collection].find({}).sort('tstamp', 1)
    df = pd.DataFrame(mydoc)
    #df.drop(['feature_importance', 'delta_minutes', 'y_check'], axis=1, inplace=True)
    df_out = df_out.append(df)

# %%
columns = ['symbol', 'interval', 'model_datetime', 'include_raw_rates',
           'include_autocorrs', 'include_all_ta', 'autocorrelation_lag',
           'autocorrelation_lag_step', 'n_rows', 'test_size', 'profit_threshold',
           'forecast_shift', 'model', 'accuracy', 'precision', 'recall',
           'n_pos_labels', 'model_how_old', 'forecast_datetime', 'y_proba_1']
df_group = df_out.groupby(['symbol', 'interval', 'forecast_datetime'], as_index=False)
df_group = df_group.max()[['symbol', 'interval', 'forecast_datetime']].values
# %%
for symbol, interval, forecast_datetime in df_group:
    df = df_out[(df_out['symbol'] == symbol) &
                (df_out['interval'] == interval) &
                (df_out['forecast_datetime'] == forecast_datetime)]

# %%
for symbol, interval, forecast_datetime in df_group:
    df = df_out[(df_out['symbol'] == symbol) &
                (df_out['interval'] == interval) &
                (df_out['forecast_datetime'] == forecast_datetime)]

# %%
for i in df['y_check'].values:
    for j in i:
        print(j)
# %%
df1.merge(df2, on='tstamp', how='outer').merge(df3, on='tstamp', how='outer').sort_values(by='tstamp')