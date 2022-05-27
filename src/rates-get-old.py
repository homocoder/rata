# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates-get-old.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=1 --tstamp=2022-05-05'
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
from rata.utils import copy_from_stringio
from rata.marketon import get_data

# %%

conn = psycopg2.connect(
    dbname='rata',
    user='rata',
    password='acaB.1312',
    host='127.0.0.1',
    port=5432,
)

tstamp = _conf['tstamp']
hours_back = 48

#%%
df = get_data.get_finnhub(symbol=_conf['symbol'], interval=_conf['interval'], exchange=exchange, kind=_conf['kind'], tstamp=tstamp, hours=hours_back)
df = df.sort_values(by='tstamp')
df.reset_index(inplace=True)

copy_from_stringio(conn, df, 'rates')
# %%
conn.commit()
conn.close()
# %%
