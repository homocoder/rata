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
import datetime
from pymongo import MongoClient
from rata.utils import sort_human

# %%
client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_forecasts_binclf']

columns = ['symbol', 'interval', 'model_datetime', 'include_raw_rates',
           'include_autocorrs', 'include_all_ta', 'autocorrelation_lag',
           'autocorrelation_lag_step', 'n_rows', 'test_size', 'profit_threshold',
           'forecast_shift', 'model', 'accuracy', 'precision', 'recall',
           'n_pos_labels', 'model_how_old', 'forecast_datetime', 'y_proba_1', 'y_check']

df_out= pd.DataFrame()

#for collection in db.list_collection_names():
for collection in ['AUDCHF_5', 'USDCAD_5']:    
    mydoc = db[collection].find({})    
    df = pd.DataFrame(mydoc)[columns]
    df_out = df_out.append(df)

# %%
df_group = df_out.groupby(['symbol', 'interval', 'forecast_datetime'], as_index=False)
df_group = df_group.max()[['symbol', 'interval', 'forecast_datetime']]
# %%
df_merge = set()
for symbol, interval, forecast_datetime in df_group.values:
    df = df_out[(df_out['symbol'] == symbol) &
                (df_out['interval'] == interval) &
                (df_out['forecast_datetime'] == forecast_datetime)]

    for i in df['y_check'].values:
        for j in i:
            row = j
            j['symbol'] = symbol
            j['interval'] = interval
            df_merge.add(str(j))

df_merge = list(df_merge)
df_merge = [eval(i) for i in df_merge]    
df_merge = pd.DataFrame(df_merge)
y_check_columns = ['tstamp', 'symbol', 'interval']
y_check_columns += sort_human(list(df_merge.columns.drop(y_check_columns).values))

df_merge = df_merge[y_check_columns]
df_merge2 = df_merge.groupby(['symbol', 'interval', 'tstamp'], as_index=False).mean()
# %%
