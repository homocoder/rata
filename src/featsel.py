# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'featsel.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3'
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
engine = create_engine('postgresql+psycopg2://rata:acab.1312@localhost:5432/rata')

symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF']

sql = "select * from feateng where symbol='" + symbols[0] + "' and interval=" + str(_conf['interval'])
df = pd.read_sql_query(sql, engine)

sql = 'with '
for s in symbols:
    sql += s + ' as (select '
    for i in df.columns:
        sql += i + ' as ' + s + '_' + i + ' ,\n'
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
df['tstamp'] = df[first_symbol.lower() + '_tstamp']

df = df[[i for i in df.columns if '_tstamp' not in i]]
df = df[[i for i in df.columns if '_symbol' not in i]]
df = df[[i for i in df.columns if '_interval' not in i]]
check_time_gaps(df, _conf)
df.set_index('tstamp', drop=True, inplace=True)
df.to_csv('audusd_gbpaud_audchf.csv')
#%%
## !!!!!!!!!!!!!!!!!!!!


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


