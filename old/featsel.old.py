# %% 🐭
from sys import argv
from xml.etree.ElementInclude import include
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

symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD'] # GBPNZD  2022-05-30 00:00:00 5085.0 # AUDNZD 2022-05-30 00:00:00 4926.0
symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'EURGBP', 'NZDUSD']

df_join = pd.DataFrame()
for s in symbols:
    sql =  "select * from feateng "
    sql += "where symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += " order by tstamp desc limit " + str(_conf['nrows'])
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    X_prefix = s + '_' + str(_conf['interval']) + '_'
    for c in df.columns:
        df.rename({c: X_prefix + c}, axis=1, inplace=True)
    
    if len(df_join) == 0:
        df_join = df
    else:
        df_join = pd.merge(df_join, df, how='inner', left_on=df_join.columns[0], right_on=df.columns[0])
df_join.sort_values(df_join.columns[0])
df_join['tstamp'] = df_join.iloc[:,0]
df_join.set_index('tstamp', drop=True, inplace=True)

check_time_gaps(df_join, {'symbol': s, 'interval': 3})
# dataframe with all data
#df_join.to_csv('../' + str(df_join.index[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(symbols) + '.' + str(len(df_join)) + '.csv')
df = df_join.copy()
len(df_join.iloc[:,0].drop_duplicates()) == len(df_join.iloc[:,0])
#%%
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)

dataset_name = str(df['tstamp'].iloc[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(symbols) + '.' + str(len(df))
dataset = dai.datasets.create(df, name=dataset_name)

ds = dai.datasets.get('f8f86436-e29d-11ec-8c63-000c291e95a2')

experiments = list()
for target_column in ['AUDUSD_3_close_SROC_' + i for i in ['12']]:
    for num_prediction_periods in [4]:
        name = target_column.replace('AUDUSD_3_close_', '') + '_FH_' + str(num_prediction_periods) + '_T_' + str(num_prediction_periods*3)
        xp = dai.experiments.create_async(train_dataset=ds,
                                            task='regression',
                                            scorer='RMSE',
                                            name=name,
                                            model='LightGBM'
                                            target_column=target_column,
                                            time_column='tstamp',
                                            num_prediction_periods=num_prediction_periods,
                                            accuracy=5,
                                            time=5,
                                            interpretability=5,
                                            config_overrides=None)
        experiments.append(xp)

# %%
xp_keys = list()
for r in dai.experiments.list().__str__().split('\n'):
    cells = r.split('|')
    if ' Experiment ' in cells:
        xp_keys.append(cells[2].strip())

df_feat_importance = pd.DataFrame()
for k in xp_keys:
    xp = dai.experiments.get(k)
    vi = xp.variable_importance()
    if vi !=  None:
        df_feat_importance = pd.concat([df_feat_importance, pd.DataFrame(vi.data, columns=vi.headers)])

featsel = df_feat_importance.groupby('description').sum().reset_index()
featsel = featsel[featsel['gain'] > 0.01].sort_values('gain', ascending=False)

# %%
symbols    = featsel['description'].str.split('_', expand=True)[[0]]
indicators = featsel['description'].str.split('_', expand=True)[[2, 3]]
SROCs      = featsel['description'].str.split('_', expand=True)[[5]]
# %%
## FEAT SELECTED 2: TOP indicators
df = df_join.copy()

top_indicators = [
    'momentum_kama',
    'momentum_pvo',
    'momentum_roc',
    'momentum_rsi',
    'momentum_stoch',
    'momentum_tsi',
    'momentum_wr',',',
    'others_cr',
    'trend_adx',
    'trend_ema',
    'trend_ichimoku',
    'trend_kst',
    'trend_mass',
    'trend_psar',
    'trend_sma',
    'trend_visual',
    'trend_vortex',
    'volatility_bbh',
    'volatility_bbl',
    'volatility_bbm',
    'volatility_bbp',
    'volatility_bbw',
    'volatility_dch',
    'volatility_dcl',
    'volatility_dcp',
    'volatility_kcc',
    'volatility_kch',
    'volatility_kcp',
    'volume_adi',
    'volume_fi',
    'volume_mfi',
    'volume_nvi',
    'volume_obv',
    'volume_sma',
    'volume_vwap'
]
## %%

top_symbols = ['AUDUSD', 'AUDCHF', 'NZDUSD']

features_selected = set()
for i in top_symbols:
    for j in df.columns:
        if (i in j):
            features_selected.add(j)
features_selected = list(features_selected)
df = df[features_selected]
df.reset_index(inplace=True)
#%%
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)

dataset_name = str(df['tstamp'].iloc[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(top_symbols) + '.' + str(len(df))
dataset = dai.datasets.create(df, name=dataset_name)

#%%
experiments = list()
for target_column in ['AUDUSD_3_close_SROC_' + i for i in ['6', '9', '12', '15']]:
    for num_prediction_periods in [2, 3, 4, 5]:
        name = target_column.replace('AUDUSD_3_close_', '') + '_H' + str(num_prediction_periods) + '_T' + str(num_prediction_periods*3)
        name = 'symsel_' + name
        xp = dai.experiments.create_async(train_dataset=dataset,
                                            task='regression',
                                            scorer='RMSE',
                                            name=name,
                                            target_column=target_column,
                                            time_column='tstamp',
                                            num_prediction_periods=num_prediction_periods,
                                            accuracy=3,
                                            time=3,
                                            interpretability=3,
                                            config_overrides=None)
        experiments.append(xp)

#%%
## FEAT SELECTED
df = df_join.copy()

top_symbols = ['AUDUSD', 'AUDCHF', 'NZDUSD']

features_selected = set()
for i in top_symbols:
    for j in df.columns:
        if (i in j):
            features_selected.add(j)
features_selected = list(features_selected)
df = df[features_selected]
df.reset_index(inplace=True)
#%%
import driverlessai

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)

dataset_name = str(df['tstamp'].iloc[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(top_symbols) + '.' + str(len(df))
dataset = dai.datasets.create(df, name=dataset_name)

#%%
experiments = list()
for target_column in ['AUDUSD_3_close_SROC_' + i for i in ['6', '9', '12', '15']]:
    for num_prediction_periods in [2, 3, 4, 5]:
        name = target_column.replace('AUDUSD_3_close_', '') + '_H' + str(num_prediction_periods) + '_T' + str(num_prediction_periods*3)
        name = 'symsel_' + name
        xp = dai.experiments.create_async(train_dataset=dataset,
                                            task='regression',
                                            scorer='RMSE',
                                            name=name,
                                            target_column=target_column,
                                            time_column='tstamp',
                                            num_prediction_periods=num_prediction_periods,
                                            accuracy=3,
                                            time=3,
                                            interpretability=3,
                                            config_overrides=None)
        experiments.append(xp)

