# %% 🐭
from re import I
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'model-dai.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3 --nrows=9000'
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
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@192.168.3.113:5432/rata')

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
df.to_csv('/home/selknam/var/' +  dataset_name + '.csv')
df.reset_index(drop=False, inplace=True)

len(df.iloc[:,0].drop_duplicates()) == len(df.iloc[:,0])
df
#%%
# FeatSel
selected_columns = [
    'tstamp',
    'AUDUSD_3_close_SROC_15',
    'AUDCHF_3_close_SROC_15',
    'NZDUSD_3_close_SROC_15',
    'AUDUSD_3_close_SROC_21',
    'AUDCHF_3_close_SROC_21',
    'NZDUSD_3_close_SROC_21',
    'AUDUSD_3_close_SROC_9',
    'AUDCHF_3_close_SROC_9',
    'NZDUSD_3_close_SROC_9']

df = df[selected_columns]

#%%
# Train / Test Datasets
df_train = df[:-800].copy()
df_test  = df[-800:].copy()
#%%
# Model TS
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)
# %%

dataset_train = dai.datasets.create(df_train, name='train.' + dataset_name)
dataset_test  = dai.datasets.create(df_test,  name='test.'  + dataset_name)

#%%
ROCs = [15]
fh = 45
expert_settings = {
    'imbalance_sampling_method': 'auto',
    'included_models': ['ImbalancedLightGBM', 'LightGBM'],
    'imbalance_sampling_threshold_min_rows_original': 1000
}
experiments = list()

for roc in ROCs:
    target_column = 'AUDUSD_3_close_SROC_' + str(roc)
    name = target_column.replace('AUDUSD_3_close_', 'regv2.') + '_FH' + str(fh) + '_T' + str(roc*3)
    xp = dai.experiments.create_async(train_dataset=dataset_train,
                                        task='regression',
                                        name=name,
                                        target_column=target_column,
                                        time_column='tstamp',
                                        num_prediction_periods=fh,
                                        accuracy=9,
                                        time=7,
                                        interpretability=1,
                                        config_overrides=None,
                                        **expert_settings)
    experiments.append(xp)

# %%
dai.experiments.list()
model = dai.experiments.get('ba752882-e68c-11ec-b938-000c291e95a2')

#%%
#Forecast
forecast = model.predict(dataset_test)

# %%
df_val = df_test.copy()
y_pred = forecast.to_pandas().iloc[:, 0]
df_val['y_pred'] = y_pred.values

y_pred_lower = forecast.to_pandas().iloc[:, 1]
df_val['y_pred_lower'] = y_pred_lower.values

y_pred_upper = forecast.to_pandas().iloc[:, 2]
df_val['y_pred_upper'] = y_pred_upper.values
df_val.to_csv('reg.csv', index=False)