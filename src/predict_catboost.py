# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'predict_catboost.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=3 --shift=1 '
fake_argv += '--X_symbols=EURUSD '#,GBPUSD '
fake_argv += '--X_include=SROC '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=3000 ' 

fake_argv += '--loss_function=MAE '

fake_argv = fake_argv.split()
#argv = fake_argv #### *!
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
    if 'interval' in c:
        ys_todelete.append(c)
    for exc in _conf['X_exclude']:
        if exc in c:
            ys_todelete.append(c)

#%%
X_predict = df[-1:]
df = df[:-_conf['shift']]
#columns containing NaNs
dfnans = pd.DataFrame(df.isnull().sum())
nancols = list(dfnans[dfnans[0] > 0].index)
ys_todelete = ys_todelete #+ nancols

X = df.drop(ys_todelete, axis=1)

if '*' in _conf['X_include']:
    ys_include = X.columns
else:
    ys_include = set()
    for c in X.columns:
        for inc in _conf['X_include']:
            if inc in c:
                ys_include.add(c)
    ys_include = list(ys_include)


X         = X[ys_include]
X_predict = X_predict[ys_include]
y         = df[_conf['y_target']]

#%%
from catboost import CatBoostRegressor

model = CatBoostRegressor(train_dir='/home/selknam/var/catboost_dir',
                          random_seed=int(datetime.datetime.now().strftime('%S%f')),
                          loss_function=_conf['loss_function'],
                          thread_count=12)

t0 = datetime.datetime.now()
model.fit(X, y)
fit_time = int((datetime.datetime.now() - t0).total_seconds())
#%% ###    SECTION ITERATE PREDICTIONS ###

for c in range(0, 90 // _conf['interval']): # 5 hours per model 
    del df_join
    del df
    del X
    del y
    from time import sleep
    while True:
        sleep(0.5)
        tnow = datetime.datetime.now()
        if (tnow.minute in [i for i in range(0, 60, _conf['interval'])]) and tnow.second == 40 + _conf['interval'] * 2: #TODO:40 hardcoded

            _conf['nrows'] = 12
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
                if 'interval' in c:
                    ys_todelete.append(c)
                for exc in _conf['X_exclude']:
                    if exc in c:
                        ys_todelete.append(c)

            #%%
            X_predict = df[-1:]
            df = df[:-_conf['shift']]
            #columns containing NaNs
            dfnans = pd.DataFrame(df.isnull().sum())
            nancols = list(dfnans[dfnans[0] > 0].index)
            ys_todelete = ys_todelete #+ nancols

            X = df.drop(ys_todelete, axis=1)

            if '*' in _conf['X_include']:
                ys_include = X.columns
            else:
                ys_include = set()
                for c in X.columns:
                    for inc in _conf['X_include']:
                        if inc in c:
                            ys_include.add(c)
                ys_include = list(ys_include)


            X         = X[ys_include]
            X_predict = X_predict[ys_include]
            y         = df[_conf['y_target']]

            y_current = y_target.replace('_shift-' + str(_conf['shift']), '').replace('_y_', '_')
            dfv  = pd.DataFrame(X_predict[y_current])
            dfv.rename({y_current: 'y_current'}, axis=1, inplace=True)
            dfv['tstamp_test'] = pd.DataFrame(y[-1:]).reset_index()['tstamp'].values[0]
            dfv['y_test'] = pd.DataFrame(y[-1:]).reset_index()[y_target].values[0]

            dfv['y_pred']    = model.predict(X_predict)
            dfv['y_proba']   = model.predict(X_predict, prediction_type='Probability')[0][0]
            dfv['symbol']    = _conf['symbol']
            dfv['interval']  = _conf['interval']
            dfv['shift']     = _conf['shift']
            dfv['X_symbols'] = ','.join(_conf['X_symbols'])

            # common to all models
            dfv['nrows']         = _conf['nrows']

            dfv['model_tstamp']  = df.index.max()
            dfv['model_id']      = str(df.index.max()).replace(' ', 'T')
            dfv['fit_time']      = fit_time

            # uncommon to models
            dfv['loss_function'] = _conf['loss_function']

            dfv.reset_index().to_sql('predict_catboost', engine, if_exists='append', index=False)
            break
# %%
