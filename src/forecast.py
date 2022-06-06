# %% 🐭
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'forecast.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3 --nrows=1200'
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
from rata.ratalib import check_time_gaps

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@localhost:5432/rata')

#symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD'] 
symbols = ['AUDUSD', 'AUDCHF', 'NZDUSD']

df_join = pd.DataFrame()
for s in symbols:
    sql =  "select * from feateng "
    sql += "where symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += " order by tstamp desc limit " + str(_conf['nrows'])
    print(sql)
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    X_prefix = s + '_' + str(_conf['interval']) + '_'
    for c in df.columns:
        df.rename({c: X_prefix + c}, axis=1, inplace=True)
    
    if len(df_join) == 0:
        df_join = df
    else:
        df_join = pd.merge(df_join, df, how='inner', left_on=df_join.columns[0], right_on=df.columns[0])

df = df_join.copy()
df.sort_values(df.columns[0])
df['tstamp'] = df.iloc[:,0]
df.sort_values('tstamp', ascending=True)
check_time_gaps(df, {'symbol': s, 'interval': 3})
df.set_index('tstamp', drop=True, inplace=True)

# dataframe with all data
dataset_name = str(df.index[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(symbols) + '.' + str(len(df))
#df.to_csv('/home/selknam/var/' +  dataset_name + '.csv')
df.reset_index(drop=False, inplace=True)

len(df.iloc[:,0].drop_duplicates()) == len(df.iloc[:,0])
#%%
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)
# %%
df_test  = df[df['tstamp'] > pd.to_datetime('2022-06-02T00:00:00')]
dataset_test = dai.datasets.create(df_test, name=dataset_name + '.v4')
model_sell = dai.experiments.get('ac287760-e4e4-11ec-9041-000c291e95a2')
model_buy  = dai.experiments.get('a92d1246-e4e4-11ec-9041-000c291e95a2')

# %%
forecast_buy  = model_buy.predict(dataset_test)
forecast_sell = model_sell.predict(dataset_test)

# %%
df_test[forecast_buy.to_pandas().columns[0]] = forecast_buy.to_pandas()[forecast_buy.to_pandas().columns[0]]
df_test[forecast_buy.to_pandas().columns[1]] = forecast_buy.to_pandas()[forecast_buy.to_pandas().columns[1]]
#%%
df_test[forecast_sell.to_pandas().columns[0]] = forecast_sell.to_pandas()[forecast_sell.to_pandas().columns[0]]
df_test[forecast_sell.to_pandas().columns[1]] = forecast_sell.to_pandas()[forecast_sell.to_pandas().columns[1]]
# %%
