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
_conf['url'] = 'postgresql+psycopg2://rata:<passwd>@' + _conf['db_host'] + ':5432/rata'

_conf

# %% Global imports
import pandas as pd
from sqlalchemy import create_engine

#%%
engine = create_engine(_conf['url'])

sql = """
select
  tstamp, y_current, interval, shift, minutes, "my_precision",
  avg(n_pred) avg_n_pred,
  avg(("pS1" + "pS2" + "pS3")/3) "avg_pS", avg(("pB1" + "pB2" + "pB3")/3) "avg_pB"
  --greatest("pS1","pS2","pS3") "pS", greatest("pB1","pB2","pB3") "pB"
from predict_clf_rf 
group by tstamp, y_current, interval, shift, minutes, "my_precision"
order by tstamp desc limit 100000
"""

with engine.connect() as conn:
    df = pd.read_sql_query(sql, conn)

df['interval']     = '__' + df['interval'].astype(str)
df['shift']        = '_' + df['shift'].astype(str)
df['minutes']      = '_' + df['minutes'].astype(str)
df['my_precision'] = df['my_precision'].str.replace('my_precision', '')
df['id_type']      = df['my_precision'].str.cat([df['minutes'], df['interval'], df['shift']])
#df = df[df['tstamp'] == '2022-10-03 14:21:00'].sort_values(by='minutes') 

list_rows = list()
for t in df['tstamp'].drop_duplicates().values:
    dft = df[df['tstamp'] == t].copy()
    dft['n_id_type']  = 'n' + dft['id_type'].copy()
    dft['pB_id_type'] = 'pB' + dft['id_type'].copy()
    dft['pS_id_type'] = 'pS' + dft['id_type'].copy()
    dp = dict()
    dp.update({'tstamp': t})
    dp.update(dft.pivot(index='tstamp', columns='n_id_type', values='avg_n_pred').to_dict(orient='records')[0])
    dp.update(dft.pivot(index='tstamp', columns='pB_id_type', values='avg_pB').to_dict(orient='records')[0])
    dp.update(dft.pivot(index='tstamp', columns='pS_id_type', values='avg_pS').to_dict(orient='records')[0])

    list_rows.append(dp)
dfpv = pd.DataFrame(list_rows)



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
