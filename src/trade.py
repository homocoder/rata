# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'trade.py --db_host=192.168.3.113 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=1 '
fake_argv += '--X_symbols=EURUSD '#,GBPUSD '
fake_argv += '--X_include=SROC '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=3000 ' 

fake_argv += '--loss_function=MAE '

fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)

_conf['X_symbols']   = _conf['X_symbols'].split(',')
_conf['X_include']   = _conf['X_include'].split(',')
_conf['X_exclude']   = _conf['X_exclude'].split(',')

y_target  = _conf['symbol'] + '_' + str(_conf['interval'])
y_target += '_y_close_SROC_' + str(_conf['shift'])
y_target += '_shift-' + str(_conf['shift'])
_conf['y_target'] = y_target

_conf

# %% Global imports
import pandas as pd
import numpy  as np
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
import datetime

engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')
#%% ###                 SECTION CREATE MODEL          ###
#%%



sql =  "select * from predict_catboost "
sql += "where symbol='" + s + "' and interval=" + str(_conf['interval'])
sql += " order by tstamp desc "
print(sql)
df = pd.read_sql_query(sql, engine).sort_values('tstamp')

#%% 
df['y_pred'].min()
# %%
