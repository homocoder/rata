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
from sqlalchemy import create_engine

# %%
engine = create_engine('postgresql+psycopg2://rata:acab.1312@localhost:5432/rata')
sql =  "select distinct symbol from rates"
df = pd.read_sql_query(sql, engine)
#%%
for symbol in df['symbol']:
    sql =  "select * from rates where symbol='" + symbol + "'"
    df = pd.read_sql_query(sql, engine)

    if len(df) == 0:
        print('No data in ', symbol)
    else:
        print('Cleaning ', symbol)
        #df['status'] = 'ok'
        df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume'], as_index=False).max()
        df = df[['query_tstamp', 'unix_tstamp', 'tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].sort_values('tstamp')

        sql =  "delete from rates where symbol='" + symbol + "'"
        engine.execute(sql)
        
        df.to_sql('rates', engine, if_exists='append', index=False)
# %%
