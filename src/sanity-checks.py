# %% 🐭
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'sanity-checks.py --db_host=localhost '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@localhost:5432/rata')
symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD']
#%%
## RATES GAPS
for s in symbols:
    sql =  "select * from rates "
    sql += "where symbol='" + s + "' and interval=1"
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    check_time_gaps(df, {'symbol': s, 'interval': 5})

#%%
## FEATENG GAPS
for s in symbols:
    sql =  "select tstamp from feateng "
    sql += "where symbol='" + s + "' and interval=3"
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    check_time_gaps(df, {'symbol': s, 'interval': 3})


# %%
