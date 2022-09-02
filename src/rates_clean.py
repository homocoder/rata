# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates_clean.py --db_host=192.168.1.83 --symbol=AUDUSD --days=1 '
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'
_conf

# %%
# Global imports
import pandas as pd
from sqlalchemy import create_engine

# %%
engine = create_engine(_conf['url'])

#%%
if _conf['symbol'] == '*':
    with engine.connect() as conn:
        sql =  "select distinct symbol from rates"
        df = pd.read_sql_query(sql, conn)
        symbols = df['symbol']
else:
    symbols = [_conf['symbol']]

for symbol in symbols:
    with engine.connect() as conn:
        sql =  "select * from rates where symbol='" + symbol + "' and tstamp > (current_date - interval '" + str(_conf['days']) + " days')"
        df = pd.read_sql_query(sql, conn)

    if len(df) == 0:
        print('No data in ', symbol)
    else:
        print('Cleaning ', symbol)

        df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume'], as_index=False).max()
        df = df[['query_tstamp', 'unix_tstamp', 'tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].sort_values('tstamp')

        with engine.connect().execution_options(autocommit=False) as conn:
            sql =  "delete from rates where symbol='" + symbol + "' and tstamp > (current_date - interval '" + str(_conf['days']) + " days')"
            tx = conn.begin()
            conn.execute(sql)
            df.to_sql('rates', conn, if_exists='append', index=False)
            tx.commit()
            tx.close()