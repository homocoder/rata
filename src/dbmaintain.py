# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'dbmaintain --db_host=localhost --db_port=5432 '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import psycopg2
from rata.utils import copy_from_stringio

# %%
## */* RATES maintenance */* ##

conn = psycopg2.connect(
    dbname='rata',
    user='rata',
    password='acab.1312',
    host=_conf['db_host'],
    port=_conf['db_port'],
)

sql =  "select distinct symbol from rates"

df = pd.read_sql_query(sql, conn)
df
#%%
for symbol in df['symbol']:

    sql =  "select * from rates where symbol='" + symbol + "'"
    df = pd.read_sql_query(sql, conn)

    if len(df) == 0:
        print('No data in ', symbol)
    else:
        print('Cleaning ', symbol)
        #df['status'] = 'ok'
        df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume'], as_index=False).max()
        df = df[['query_tstamp', 'unix_tstamp', 'tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].sort_values('tstamp')

        cur = conn.cursor()
        sql =  "delete from rates where symbol='" + symbol + "'"
        cur.execute(sql)
        
        copy_from_stringio(conn, df, 'rates') #TODO: error

conn.commit()
cur.close()
conn.close()
# %%
