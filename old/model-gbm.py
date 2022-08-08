# %% 🐭
from sys import argv
from rata.utils import parse_argv, split_sequences

fake_argv  = 'model-gbm.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=3 '
fake_argv += '--X_symbols=EURUSD,AUDUSD,GBPUSD '
fake_argv += '--X_include=vpt,rsi,stoch,others_cr,macd,kst,adx,cci,dch,* '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=3000 ' 
fake_argv += '--test_lenght=800 '

fake_argv += '--iterations=30 '
fake_argv += '--learning_rate=0.3 '
fake_argv += '--depth=3 '
fake_argv += '--l2_leaf_reg=3 '
fake_argv += '--loss_function=RMSE '

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
from sklearn.metrics import mean_squared_error

#%%
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')

df_join = pd.DataFrame()
for s in _conf['X_symbols']:
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
        print('Merge')
        df_join = pd.merge(df_join, df, how='inner', left_on=df_join.columns[0], right_on=df.columns[0])

df = df_join.copy()
df.sort_values(df.columns[0])
df['tstamp'] = df.iloc[:,0]
df.sort_values('tstamp', ascending=True)
check_time_gaps(df, {'symbol': s, 'interval': 3})
df.set_index('tstamp', drop=True, inplace=True)

print(len(df.iloc[:,0].drop_duplicates()) == len(df.iloc[:,0]))
df.sort_values('tstamp', inplace=True)
df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

#%% Features handling
ys_todelete = list()
for c in df.columns:
    if 'shift' in c:
        ys_todelete.append(c)
    if 'symbol' in c:
        ys_todelete.append(c)

df = df[:-9]
#columns containing NaNs
dfnans = pd.DataFrame(df.isnull().sum())
nancols = list(dfnans[dfnans[0] > 0].index)

X = df.drop(ys_todelete + ['EURUSD_1_symbol'] + nancols, axis=1)

y = df[_conf['y_target']]
X_train = X[:-_conf['test_lenght']]
y_train = y[:-_conf['test_lenght']]

X_test = X[-_conf['test_lenght']:]
y_test = y[-_conf['test_lenght']:]

#%%
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=_conf['iterations'], 
                          depth=_conf['depth'], 
                          learning_rate=_conf['learning_rate'], 
                          loss_function=_conf['loss_function'])

t0 = datetime.datetime.now()
model.fit(X_train, y_train)
fit_time = int((datetime.datetime.now() - t0).total_seconds() / 60)
#%%
dfv  = pd.DataFrame(y_test)
dfv.rename({y_target: 'y_test'}, axis=1)
dfv['y_pred']    = model.predict(X_test)
dfv['symbol']    = _conf['symbol']
dfv['interval']  = _conf['interval']
dfv['shift']     = _conf['shift']
dfv['X_symbols'] = ','.join(_conf['X_symbols'])

dfv['test_lenght']  = _conf['test_lenght']
dfv['nrows']        = _conf['nrows']

dfv['model_tstamp'] = df.index.max()
dfv['model_id']     = str(df.index.max()).replace(' ', 'T')
dfv['fit_time']     = fit_time

dfv['iterations']    = _conf['iterations']
dfv['learning_rate'] = _conf['learning_rate']
dfv['depth']         = _conf['depth']
dfv['l2_leaf_reg']   = _conf['l2_leaf_reg']

dfv['mse']          = mean_squared_error(dfv['y_test'], dfv['y_pred'])
# %% Feature importance
dffi = pd.DataFrame(model.feature_names_)
dffi.rename({0: 'feature_name'}, axis=1, inplace=True)
dffi['feature_importance'] = model.feature_importances_
dffi.sort_values('feature_importance', ascending=False, inplace=True)
dffi.head(45)

#%%
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')
df_val.reset_index().to_sql('catboost', engine, if_exists='append', index=False)

# %%
df_val
# %%
