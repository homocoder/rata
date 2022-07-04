# %% 🐭
from sys import argv
from rata.utils import lstm_prep, parse_argv

fake_argv = 'model-flaml.py --db_host=localhost --symbol=AUDUSD --kind=forex --interval=3 --nrows=6500'
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
#df.to_csv('/home/selknam/var/' +  dataset_name + '.csv')
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
from flaml import AutoML
automl = AutoML()

# configure AutoML settings
settings = {
    "time_budget": 120,  # total running time in seconds
    "metric": "rmse",  # primary metric
    "task": "ts_forecast",  # task type
    "log_file_name": "rata.audusd.flaml.log",  # flaml log file
    "eval_method": "holdout",
    "log_type": "all",
    "label": "AUDUSD_3_close_SROC_15",
    "estimator_list": ['lgbm', 'xgboost']
}
time_horizon = 15

# train the model
automl.fit(dataframe=df_train.dropna(), **settings, period=time_horizon)

# %%
df_pred = df_test.copy()
df_pred['y_pred'] = automl.predict(df_test)



#%%





#%%
tmp = [
    'AUDCHF_3_momentum_pvo_hist_SROC_21',
    'AUDCHF_3_momentum_pvo_signal_SROC_21',
    'AUDCHF_3_trend_adx',
    'AUDCHF_3_trend_aroon_down_SROC_21',
    'AUDCHF_3_trend_mass_index_SROC_21',
    'AUDCHF_3_trend_psar_up',
    'AUDCHF_3_trend_trix_SROC_15',
    'AUDCHF_3_trend_visual_ichimoku_a',
    'AUDCHF_3_trend_visual_ichimoku_b',
    'AUDCHF_3_volatility_bbli_SROC_21',
    'AUDCHF_3_volume_nvi',
    'AUDCHF_3_volume_obv_SROC_18',
    'AUDUSD_3_momentum_pvo_hist_SROC_6',
    'AUDUSD_3_trend_adx_SROC_21',
    'AUDUSD_3_trend_sma_slow',
    'AUDUSD_3_volatility_bbl',
    'AUDUSD_3_volatility_dcl',
    'NZDUSD_3_momentum_pvo_signal_SROC_9',
    'NZDUSD_3_trend_adx_SROC_6',
    'NZDUSD_3_trend_aroon_down_SROC_15',
    'NZDUSD_3_trend_kst_diff_SROC_18',
    'NZDUSD_3_trend_psar_down_indicator_SROC_21',
    'NZDUSD_3_trend_visual_ichimoku_b_SROC_21',
    'NZDUSD_3_volatility_dcw_SROC_21',
    'NZDUSD_3_volume',
    'NZDUSD_3_volume_nvi_SROC_12'
]
# %%
import numpy as np
# checking for infinity
print()
print("checking for infinity")
  
ds = df.isin([np.inf, -np.inf])
print(ds)
  
# printing the count of infinity values
print()
print("printing the count of infinity values")
  

  
# counting infinity in a particular column name
for column in df.columns:
    print(column)
    c = np.isinf(df[column]).values.sum()
    print("It contains " + str(c) + " infinite values")
  
# printing column name where infinity is present
print()
print("printing column name where infinity is present")
col_name = df.columns.to_series()[np.isinf(df).any()]
print(col_name)
  
# printing row index with infinity
print()
print("printing row index with infinity ")
  
r = df.index[np.isinf(df).any(1)]
print(r)
# %%
from hcrystalball.utils import get_sales_data
import numpy as np
from flaml import AutoML

time_horizon = 30
df = get_sales_data(n_dates=180, n_assortments=1, n_states=1, n_stores=1)
df = df[["Sales", "Open", "Promo", "Promo2"]]

# feature engineering - create a discrete value column
# 1 denotes above mean and 0 denotes below mean
df["above_mean_sales"] = np.where(df["Sales"] > df["Sales"].mean(), 1, 0)
df.reset_index(inplace=True)

# train-test split
discrete_train_df = df[:-time_horizon]
discrete_test_df = df[-time_horizon:]
discrete_X_train, discrete_X_test = (
    discrete_train_df[["Date", "Open", "Promo", "Promo2"]],
    discrete_test_df[["Date", "Open", "Promo", "Promo2"]],
)
discrete_y_train, discrete_y_test = discrete_train_df["above_mean_sales"], discrete_test_df["above_mean_sales"]

# initialize AutoML instance
automl = AutoML()

# configure the settings
settings = {
    "time_budget": 15,  # total running time in seconds
    "metric": "accuracy",  # primary metric
    "task": "ts_forecast_classification",  # task type
    "log_file_name": "sales_classification_forecast.log",  # flaml log file
    "eval_method": "holdout",
}

# train the model
automl.fit(X_train=discrete_X_train,
           y_train=discrete_y_train,
           **settings,
           period=time_horizon)

# make predictions
discrete_y_pred = automl.predict(discrete_X_test)
print("Predicted label", discrete_y_pred)
print("True label", discrete_y_test)