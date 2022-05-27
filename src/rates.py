# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=1 '
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
if _conf['kind'] == 'forex':
    exchange = 'OANDA'
if _conf['kind'] == 'crypto':
    exchange = 'COINBASE'
if ':' in _conf['symbol']:
    exchange        = _conf['symbol'].split(':')[0]
    _conf['symbol'] = _conf['symbol'].split(':')[1]
_conf

# %%
# Global imports
import psycopg2
import datetime as dt
from rata.marketon import get_data
from random import random
from time import sleep
from rata.utils import copy_from_stringio

# %%

conn = psycopg2.connect(
    dbname='rata',
    user='rata',
    password='acaB.1312',
    host='127.0.0.1',
    port=5432,
)

cur = conn.cursor()
sql =  "select max(tstamp) from rates r "
sql += "where r.symbol='" + _conf['symbol'] + "' and r.interval=" + str(_conf['interval'])
cur.execute(sql)
query = cur.fetchone()

if query[0] == None:
    hours_back = 80 * _conf['interval']
else:
    t1 = query[0]
    t2 = dt.datetime.utcnow()
    t3 = t2 - t1
    hours_back = (t3.seconds // 3600) + 1

print('Hours back: ', hours_back)

#%%
sleep(random() * 1)

df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], exchange=exchange, kind=_conf['kind'], hours=hours_back)
df = df.sort_values(by='tstamp')
df.reset_index(inplace=True)

copy_from_stringio(conn, df, 'rates')
# %%
conn.commit()
cur.close()
conn.close()