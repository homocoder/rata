# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'sanity_checks.py --db_host=192.168.1.84 '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@192.168.1.84:5432/rata')
sql =  "select distinct symbol from rates"
df = pd.read_sql_query(sql, engine)
symbols = df['symbol']

#%%
## RATES GAPS 1
for s in symbols:
    sql =  "select * from rates "
    sql += "where symbol='" + s + "' and interval=1"
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    check_time_gaps(df, {'symbol': s, 'interval': 4})
    print('Count duplicates ', s, len(df['tstamp']) - len(df['tstamp'].drop_duplicates()))
    
#%%
## FEATENG GAPS 1
for s in symbols:
    sql =  "select tstamp from feateng "
    sql += "where symbol='" + s + "' and interval=1"
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    check_time_gaps(df, {'symbol': s, 'interval': 3})
    print('Count duplicates ', s, len(df['tstamp']) - len(df['tstamp'].drop_duplicates()))
    
# %%
## FEATENG GAPS 3
for s in symbols:
    sql =  "select tstamp from feateng "
    sql += "where symbol='" + s + "' and interval=3"
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    check_time_gaps(df, {'symbol': s, 'interval': 3})
    print('Count duplicates ', s, len(df['tstamp']) - len(df['tstamp'].drop_duplicates()))
# %%