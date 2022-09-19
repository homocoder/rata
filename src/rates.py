# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates.py --db_host=192.168.1.84 --symbol=EURUSD --kind=forex --interval=1 '
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)

if _conf['kind'] == 'forex':
    exchange = 'OANDA'
if _conf['kind'] == 'crypto':
    exchange = 'COINBASE'
if ':' in _conf['symbol']:
    exchange        = _conf['symbol'].split(':')[0]
    _conf['symbol'] = _conf['symbol'].split(':')[1]
_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'
if not('tstamp' in _conf.keys()):
    _conf['tstamp'] = None
_conf

# %% Global imports
import datetime as dt
from rata.marketon import get_data
from random import random
from time import sleep
from sqlalchemy import create_engine

# %% Query last rate on DB
from pandas import read_sql_query
engine = create_engine(_conf['url'])
with engine.connect().execution_options(autocommit=False) as conn:
    sql =  "select max(tstamp) from rates r "
    sql += "where r.symbol='" + _conf['symbol'] + "' and r.interval=" + str(_conf['interval'])
    dfq = read_sql_query(sql, conn)
    query_max = dfq.iloc[0, 0]

#%% Calculate hours_back
if query_max is None:
    hours_back = 50 * _conf['interval']
elif not(_conf['tstamp'] == None):
    hours_back = 48
else:
    t1 = query_max
    t2 = dt.datetime.utcnow()
    t3 = t2 - t1
    hours_back = (t3.total_seconds() // 3600) + 1

# OANDA limit is 50  hours at interval=1, 2952 rows
# OANDA limit is 250 hours at interval=3, 2415 rows
#hours_back = 200
print('Hours back: ', hours_back)

#%% Connect to API
sleep(random() * 2)
df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], exchange=exchange, kind=_conf['kind'], tstamp=_conf['tstamp'], hours=hours_back)
df = df.sort_values(by='tstamp')
df.reset_index(inplace=True)

#%% Update last rates to DB
from sqlalchemy import create_engine
engine = create_engine(_conf['url'])
with engine.connect().execution_options(autocommit=False) as conn:
    tx = conn.begin()
    df.to_sql('rates', conn, if_exists='append', index=False)
    tx.commit()
    tx.close()
# %%
