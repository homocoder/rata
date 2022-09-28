# %% 🐭
from sys import argv
from this import d
from rata.utils import parse_argv

fake_argv  = 'trade.py --db_host=192.168.1.84 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=1 '
fake_argv += '--X_symbols=EURUSD '#,GBPUSD '
fake_argv += '--X_include=SROC '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)

_conf['X_symbols']   = _conf['X_symbols'].split(',')
_conf['X_include']   = _conf['X_include'].split(',')
_conf['X_exclude']   = _conf['X_exclude'].split(',')

y_target  = _conf['symbol'] + '_' + str(_conf['interval'])
y_target += '_y_close_SROC_' + str(_conf['shift'])
y_target += '_shift-' + str(_conf['shift'])
_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'

_conf

# %% Global imports
import pandas as pd
from sqlalchemy import create_engine

#%%
engine = create_engine(_conf['url'])

sql = """
select
  tstamp, y_current, interval, shift, minutes, "my_test_precisionS", n_pred, 
  "pS1"+"pS2"+"pS3" "pS"
from predict_clf_rf 
where
  "my_test_precisionS" > 0.8
order by tstamp desc
"""

with engine.connect() as conn:
    df = pd.read_sql_query(sql, conn)

# %%
df[df['tstamp'] == '2022-09-27 22:15:00'].sort_values(by='minutes')
# %%
q = 0.0
df11 = df[(df['interval'] == 1) & (df['shift'] == 1)]
df11 = df11[df11['y_pred'] > df11['y_pred'].quantile(q)]
df13 = df[(df['interval'] == 1) & (df['shift'] == 3)]
df13 = df13[df13['y_pred'] > df13['y_pred'].quantile(q)]
df16 = df[(df['interval'] == 1) & (df['shift'] == 6)]
df16 = df16[df16['y_pred'] > df16['y_pred'].quantile(q)]
df19 = df[(df['interval'] == 1) & (df['shift'] == 9)]
df19 = df19[df19['y_pred'] > df19['y_pred'].quantile(q)]
df31 = df[(df['interval'] == 3) & (df['shift'] == 1)]
df31 = df31[df31['y_pred'] > df31['y_pred'].quantile(q)]
df33 = df[(df['interval'] == 3) & (df['shift'] == 3)]
df33 = df33[df33['y_pred'] > df33['y_pred'].quantile(q)]
df36 = df[(df['interval'] == 3) & (df['shift'] == 6)]
df36 = df36[df36['y_pred'] > df36['y_pred'].quantile(q)]
df39 = df[(df['interval'] == 3) & (df['shift'] == 9)]
df39 = df39[df39['y_pred'] > df39['y_pred'].quantile(q)]
dfB = pd.concat([df11, df13, df16, df19, df31, df33, df39]).sort_index()
q = 1.0
df11 = df[(df['interval'] == 1) & (df['shift'] == 1)]
df11 = df11[df11['y_pred'] < df11['y_pred'].quantile(q)]
df13 = df[(df['interval'] == 1) & (df['shift'] == 3)]
df13 = df13[df13['y_pred'] < df13['y_pred'].quantile(q)]
df16 = df[(df['interval'] == 1) & (df['shift'] == 6)]
df16 = df16[df16['y_pred'] < df16['y_pred'].quantile(q)]
df19 = df[(df['interval'] == 1) & (df['shift'] == 9)]
df19 = df19[df19['y_pred'] < df19['y_pred'].quantile(q)]
df31 = df[(df['interval'] == 3) & (df['shift'] == 1)]
df31 = df31[df31['y_pred'] < df31['y_pred'].quantile(q)]
df33 = df[(df['interval'] == 3) & (df['shift'] == 3)]
df33 = df33[df33['y_pred'] < df33['y_pred'].quantile(q)]
df36 = df[(df['interval'] == 3) & (df['shift'] == 6)]
df36 = df36[df36['y_pred'] < df36['y_pred'].quantile(q)]
df39 = df[(df['interval'] == 3) & (df['shift'] == 9)]
df39 = df39[df39['y_pred'] < df39['y_pred'].quantile(q)]
dfS = pd.concat([df11, df13, df16, df19, df31, df33, df39]).sort_index()
# %%
dfB
# %%
dfS
# %%
df_pred = df[['y_pred', 'symbol', 'interval', 'shift']].reset_index()
df_test = df[['tstamp_test', 'y_current', 'symbol', 'interval', 'shift']].reset_index(drop=True).rename({'tstamp_test': 'tstamp'}, axis=1)
df_pred.merge(df_test, on=['tstamp', 'symbol', 'interval', 'shift'], how='inner')#.sort_index().tail(60)
# %%
df_pred
#df_test

# %%
df_test
# %%
df11 = df[(df['interval'] == 1) & (df['shift'] == 1)]
df13 = df[(df['interval'] == 1) & (df['shift'] == 3)]
df16 = df[(df['interval'] == 1) & (df['shift'] == 6)]
df19 = df[(df['interval'] == 1) & (df['shift'] == 9)]
df31 = df[(df['interval'] == 3) & (df['shift'] == 1)]
df33 = df[(df['interval'] == 3) & (df['shift'] == 3)]
df36 = df[(df['interval'] == 3) & (df['shift'] == 6)]
df39 = df[(df['interval'] == 3) & (df['shift'] == 9)]
df11

# %%
