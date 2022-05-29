# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'feateng.py --db_host=localhost --symbol=NZDUSD --kind=forex --interval=3'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import custom_resample_close, custom_resample_open, custom_resample_volume, check_time_gaps

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@localhost:5432/rata')

sql =  "with a as ("
sql += "  select distinct tstamp from rates r1 "
sql += "  where r1.symbol='" + _conf['symbol'] + "' and r1.interval=1 "
sql += "  order by r1.tstamp desc limit " + str(_conf['interval'] * 9000) + "),"
sql += "b as (select min(tstamp) from a)"
sql += "select * from rates r2 where tstamp > (select * from b)"
sql += "and r2.symbol='" + _conf['symbol'] + "' and r2.interval=1 "

df = pd.read_sql_query(sql, engine)
#%%
df.sort_values(by='tstamp', ascending=True, inplace=True)
symbol    = df[['symbol']].iloc[0]['symbol']
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

tmp = check_time_gaps(df, _conf)
# %% 🐭
# Technical Indicators
import ta
df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
# TODO: SROC3 SROC6 SROC9 SROC12
# TODO: SROC3 SROC6 SROC9 SROC12
#%%
sql =  "delete from feateng where symbol='" + _conf['symbol'] + "' and interval=" + str(_conf['interval'])
engine.execute(sql)
df.to_sql('feateng', engine, if_exists='append', index=False)