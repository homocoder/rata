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
from rata.ratalib import custom_resample_close, custom_resample_open, custom_resample_volume, check_time_gaps

#%%
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@localhost:5432/rata')

symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD']

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
df_join.to_csv('../' + str(df_join.index[-1]).replace(' ', 'T').replace(':', '-') + '.' + '_'.join(symbols) + '.' + str(len(df_join)) + '.csv')
len(df_join.iloc[:,0].drop_duplicates()) == len(df_join.iloc[:,0])
#%%
import driverlessai
import matplotlib.pyplot as plt
import pandas as pd

address = 'http://192.168.3.114:12345'
username = 'admin'
password = 'admin'
dai = driverlessai.Client(address = address, username = username, password = password)

ds = dai.datasets.get('f8f86436-e29d-11ec-8c63-000c291e95a2')

experiments = list()
for target_column in ['AUDUSD_3_close_SROC_' + i for i in ['3', '6', '9', '12', '15']]:
    for num_prediction_periods in [3, 4]:
        name = target_column.replace('AUDUSD_3_close_', '') + '_FH_' + str(num_prediction_periods) + '_T_' + str(num_prediction_periods*3)
        xp = dai.experiments.create_async(train_dataset=ds,
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
#SROC_3 = chao
top_indicators = momentum_kama
momentum_pvo
momentum_roc
momentum_rsi
momentum_stoch
momentum_tsi
momentum_wr
others_cr
trend_adx
trend_ema
trend_ichimoku
trend_kst
trend_mass
trend_psar
trend_sma
trend_visual
trend_vortex
volatility_bbh
volatility_bbl
volatility_bbm
volatility_bbp
volatility_bbw
volatility_dch
volatility_dcl
volatility_dcp
volatility_kcc
volatility_kch
volatility_kcp
volume_adi
volume_fi
volume_mfi
volume_nvi
volume_obv
volume_sma
volume_vwap

top_symbols
AUDUSD
AUDCHF
NZDUSD
AUDNZD
GBPAUD



