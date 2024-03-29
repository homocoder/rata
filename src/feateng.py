# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'feateng.py --db_host=192.168.1.84 --symbol=USDJPY --kind=forex --interval=3 --nrows=350 ' #nrows always >= 300
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
#argv = "feateng.py --db_host=192.168.1.84 --symbol=AUDUSD --kind=forex --interval=1 --nrows=350".split()
_conf = parse_argv(argv=argv)
_conf['url'] = 'postgresql+psycopg2://rata:<passwd>@' + _conf['db_host'] + ':5432/rata'
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import custom_resample_close, custom_resample_open, custom_resample_volume, check_time_gaps

#%%
from sqlalchemy import create_engine
engine = create_engine(_conf['url'])

sql =  "with a as ("
sql += "  select distinct tstamp from rates r1 "
sql += "  where r1.symbol='" + _conf['symbol'] + "' and r1.interval=1 "
sql += "  order by r1.tstamp desc limit " + str(_conf['interval'] * _conf['nrows']) + ")," 
sql += "b as (select min(tstamp) from a) "
sql += "select * from rates r2 where tstamp > (select * from b)"
sql += "and r2.symbol='" + _conf['symbol'] + "' and r2.interval=1 "

with engine.connect() as conn:
    df = pd.read_sql_query(sql, conn)

df.sort_values(by='tstamp', ascending=True, inplace=True)
symbol    = df[['symbol']].iloc[0]['symbol']
interval  = int(df[['interval']].iloc[0]['interval']) # always 1

# %%
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

tmp = check_time_gaps(df, _conf)
# %% 🐭
# Technical Indicators
import ta
df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

for c in df.columns.drop(['tstamp', 'symbol', 'interval']):
    for i in [6, 9, 15, 30, 60, 90]:
        df[str(c) + '_SROC_' + str(i)] = df[c].pct_change(i) * 100
        df[str(c) + '_SROC_' + str(i)] = df[str(c) + '_SROC_' + str(i)].rolling(window=i).mean()
df = df[200:] # always > 180
print(len(df))
#%%
for shift in ['6', '9', '15', '30', '60', '90']:
    # Ys for regression
    df['y_close_SROC_' + shift + '_shift-' + shift] = df['close_SROC_' + shift].shift(-(int(shift)))
    # Ys for classification
    # SROC_1 BUY
    df['y_B_close_SROC_' + shift + '_shift-' + shift] = 0
    df['y_B_close_SROC_' + shift + '_shift-' + shift] = df['y_B_close_SROC_' + shift + '_shift-' + shift].mask(df['y_close_SROC_' + shift + '_shift-' + shift] >  0.025, 1)
    # SROC_1 SELL
    df['y_S_close_SROC_' + shift + '_shift-' + shift] = 0
    df['y_S_close_SROC_' + shift + '_shift-' + shift] = df['y_S_close_SROC_' + shift + '_shift-' + shift].mask(df['y_close_SROC_' + shift + '_shift-' + shift] < -0.025, 1)

#%% Check if table exists
sql  = "SELECT COUNT(table_name) FROM information_schema.tables "
sql += " WHERE table_schema LIKE 'public' AND table_type LIKE 'BASE TABLE' AND "
sql += "table_name = 'feateng' "

with engine.connect() as conn:
    table_exists = pd.read_sql_query(sql, conn).iloc[0, 0]
    
sql  = "delete from feateng where symbol='" + _conf['symbol']
sql += "' and interval=" + str(_conf['interval'])
sql += " and tstamp >= '" + min(df['tstamp']).isoformat() + "'::timestamp "

with engine.connect().execution_options(autocommit=False) as conn:
    tx = conn.begin()
    if table_exists:
        conn.execute(sql)
    df.to_sql('feateng', engine, if_exists='append', index=False)
    tx.commit()
    tx.close()
# %%
