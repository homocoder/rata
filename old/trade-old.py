# %% 🐭
from sys import argv
from this import d
from rata.utils import parse_argv

fake_argv  = 'trade.py --db_host=192.168.1.84 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=1 '
fake_argv += '--X_symbols=EURUSD '#,GBPUSD '
fake_argv += '--X_include=SROC '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=3000 ' 

fake_argv += '--loss_function=MAE '

fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)

_conf['X_symbols']   = _conf['X_symbols'].split(',')
_conf['X_include']   = _conf['X_include'].split(',')
_conf['X_exclude']   = _conf['X_exclude'].split(',')

y_target  = _conf['symbol'] + '_' + str(_conf['interval'])
y_target += '_y_close_SROC_' + str(_conf['shift'])
y_target += '_shift-' + str(_conf['shift'])
_conf['y_target'] = y_target

_conf

# %% Global imports
import pandas as pd
import numpy  as np
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
import datetime

engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')
#%%
sql =  "select * from predict_catboost "
#sql += "where symbol='" + _conf['symbol']+ "' and interval=" + str(_conf['interval'])
sql += " where tstamp > (now() - interval '12 hours')"
sql += " order by tstamp desc "

sql = """
with
p as
(select tstamp, symbol, "interval", shift, avg(y_pred) y_pred_avg
from predict_catboost 
group by  tstamp, symbol, "interval", shift
order by tstamp desc),
t as
(select tstamp_test, avg(y_current) y_current_avg, y_test, symbol, "interval", shift 
from predict_catboost
--where y_current = y_test --NaNs ??? :(
group by tstamp_test, y_test, symbol, "interval", shift
order by tstamp_test desc )
select p.tstamp, t.tstamp_test, y_current_avg, y_test, y_pred_avg, t.symbol, t.interval, t.shift
from p inner join t
  on 
    p.tstamp = t.tstamp_test and
    p.symbol = t.symbol and
    p.interval = t.interval and
    p.shift = t.shift
where tstamp > (now() - interval '24 hours')
order by tstamp desc
"""

print(sql)
df = pd.read_sql_query(sql, engine).sort_values('tstamp')
df.set_index('tstamp', inplace=True)
df
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
