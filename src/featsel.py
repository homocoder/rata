# %% 🐭
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'featsel.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3 --nrows=6000'
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
#TRAIN

# Y for regression
y_shifted  = df['AUDUSD_3_close_SROC_15'].shift(-15)

# Y for classification
df['YB_AUDUSD_3_close_SROC_15'] = 0
df['YB_AUDUSD_3_close_SROC_15'] = df['YB_AUDUSD_3_close_SROC_15'].mask(y_shifted >  0.2, 1)

df['YS_AUDUSD_3_close_SROC_15'] = 0
df['YS_AUDUSD_3_close_SROC_15'] = df['YS_AUDUSD_3_close_SROC_15'].mask(y_shifted < -0.2, 1)

df = df[:-15]
df_train = df[df['tstamp'] < pd.to_datetime('2022-06-02T00:00:00')]
df_test  = df[df['tstamp'] > pd.to_datetime('2022-06-02T00:00:00')]
#%%
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)

dataset_test       = dai.datasets.create(df_test, name='v2.test.'  + dataset_name)

dataset_train_buy  = dai.datasets.create(df_train.drop('YS_AUDUSD_3_close_SROC_15', axis=1), name='v2.buy.'  + dataset_name)
dataset_train_sell = dai.datasets.create(df_train.drop('YB_AUDUSD_3_close_SROC_15', axis=1), name='v2.sell.' + dataset_name)

# %%
ROCs = [15]
fh = 1
expert_settings = {
    'imbalance_sampling_method': 'auto',
    'included_models': ['ImbalancedLightGBM', 'LightGBM'],
    'imbalance_sampling_threshold_min_rows_original': 1000
}
experiments = list()
#BUY
for roc in ROCs:
    target_column = 'YB_AUDUSD_3_close_SROC_' + str(roc)
    name = target_column.replace('AUDUSD_3_close_', 'v3.buy.') + '_FH' + str(fh) + '_T' + str(roc*3)
    xp = dai.experiments.create_async(train_dataset=dataset_train_buy,
                                        task='classification',
                                        name=name,
                                        target_column=target_column,
                                        time_column='tstamp',
                                        num_prediction_periods=fh,
                                        accuracy=10,
                                        time=7,
                                        interpretability=4,
                                        config_overrides=None,
                                        **expert_settings)
    experiments.append(xp)

#SELL
for roc in ROCs:
    target_column = 'YS_AUDUSD_3_close_SROC_' + str(roc)
    name = target_column.replace('AUDUSD_3_close_', 'v3.sell.') + '_FH' + str(fh) + '_T' + str(roc*3)
    xp = dai.experiments.create_async(train_dataset=dataset_train_sell,
                                        task='classification',
                                        name=name,
                                        target_column=target_column,
                                        time_column='tstamp',
                                        num_prediction_periods=fh,
                                        accuracy=10,
                                        time=7,
                                        interpretability=4,
                                        config_overrides=None,
                                        **expert_settings)
    experiments.append(xp)
# %%
#FORECAST