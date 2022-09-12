# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'predict_clf_rf.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=9 '
fake_argv += '--X_symbols=EURUSD,NZDUSD '
fake_argv += '--X_include=close,rsi '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=9000 ' 
fake_argv += '--tstamp=2023-08-18T21:51:00 ' 
fake_argv += '--nbins=9 '

fake_argv += '--n_estimators=100 '
fake_argv += '--bootstrap=False '
fake_argv += '--class_weight=balanced_subsample '
fake_argv += '--n_jobs=2 '
fake_argv += '--my_test_precisionB=0.0 '
fake_argv += '--my_test_precisionS=0.0 '
fake_argv += '--my_moving_precisionB=0.0 '
fake_argv += '--my_moving_precisionS=0.0 '

fake_argv = fake_argv.split()
#argv = fake_argv #### *!
argv="python3 -u --db_host=192.168.1.83 --symbol=EURUSD --interval=1 --shift=90 --X_symbols=EURUSD,NZDUSD --X_include=atr,vpt,rsi,stoch,others_cr,macd,kst,adx,cci,dch,open,high,low,close,volume,obv --X_exclude=volatility_kcli --tstamp=2023-01-01T00:00:00 --nrows=7000 --test_lenght=800 --nbins=12 --n_estimators=300 --bootstrap=False --class_weight=None --n_jobs=2 --random_state=11715967".split()

_conf = parse_argv(argv=argv)
_conf['n_jobs'] = 2
_conf['X_symbols']   = _conf['X_symbols'].split(',')
_conf['X_include']   = _conf['X_include'].split(',')
_conf['X_exclude']   = _conf['X_exclude'].split(',')

if _conf['class_weight'] == 'None':
    _conf['class_weight'] = None

y_target  = _conf['symbol'] + '_' + str(_conf['interval'])
y_target += '_y_close_SROC_' + str(_conf['shift'])
y_target += '_shift-' + str(_conf['shift'])
_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'
_conf['y_target'] = y_target

_conf

# %% Global imports
import pandas as pd
import numpy  as np
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
import datetime

#%%

engine = create_engine(_conf['url'])

df_join = pd.DataFrame()
for s in _conf['X_symbols']:
    sql =  "select * from feateng "
    sql += "where tstamp < '" + str(_conf['tstamp']) + "' and "
    sql += " symbol='" + s + "' and interval=" + str(_conf['interval'])
    sql += " order by tstamp desc limit " + str(_conf['nrows'])
    print(sql)
    df = pd.read_sql_query(sql, engine).sort_values('tstamp')
    X_prefix = s + '_' + str(_conf['interval']) + '_'
    for c in df.columns:
        df.rename({c: X_prefix + c}, axis=1, inplace=True)
    
    if len(df_join) == 0:
        df_join = df
    else:
        print("To Join")
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
    if 'tstamp' in c:
        ys_todelete.append(c)
    for exc in _conf['X_exclude']:
        if exc in c:
            ys_todelete.append(c)

X_predict = df[-1:]
df = df[:-_conf['shift']]
#columns containing NaNs
dfnans = pd.DataFrame(df.isnull().sum())
nancols = list(dfnans[dfnans[0] > 0].index)
ys_todelete = ys_todelete + nancols
print(nancols) # TODO: check nancols len > 0
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

X = X[ys_include]
X['hour'] = X.index.hour.values
X_predict = X_predict[ys_include]
X_predict['hour'] = X_predict.index.hour.values
y = df[_conf['y_target']]

bins = np.linspace(0, 1, _conf['nbins'] + 1)
yc = pd.qcut(y, bins)
cat_list = yc.cat.categories.astype(str).str.replace('(', 'cat_(').to_list()
yc = yc.astype(str).str.replace('(', 'cat_(')
cat_dict = dict()
for n in range(0, len(cat_list)):
    cat_dict[cat_list[n]] = 'cl_' + str(n).zfill(2)

yc = yc.map(cat_dict)
cl_list = yc.drop_duplicates().sort_values().to_list()

del y
y = yc.copy()
    
#%%
from sklearn.ensemble import RandomForestClassifier
if 'random_state' in _conf:
    random_state = _conf['random_state']
else:
    random_state = int(datetime.datetime.now().strftime('%S%f'))

model = RandomForestClassifier( n_estimators=_conf['n_estimators'],
                                random_state=random_state,
                                class_weight=_conf['class_weight'],
                                bootstrap=_conf['bootstrap'],
                                n_jobs=_conf['n_jobs'])

t0 = datetime.datetime.now()
model.fit(X, y)
fit_time = (datetime.datetime.now() - t0).total_seconds() 

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
        if (tnow.minute in [i for i in range(0, 60, _conf['interval'])]) and tnow.second == 3 + _conf['interval'] * 2: #TODO:30 hardcoded

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

            X_predict = df[-1:]

            #X         = X[ys_include]
            X_predict = X_predict[ys_include]
            #X['hour'] = X.index.hour.values
            X_predict['hour'] = X_predict.index.hour.values

            y_current = y_target.replace('_shift-' + str(_conf['shift']), '').replace('_y_', '_')
            dfr  = pd.DataFrame(X_predict[y_current])
            dfr.rename({y_current: 'y_current'}, axis=1, inplace=True)

            dfr['symbol']    = _conf['symbol']
            dfr['interval']  = _conf['interval']
            dfr['shift']     = _conf['shift']
            dfr['minutes']   = _conf['interval'] * _conf['shift']
            dfr['my_test_precisionB']   = 0.0
            dfr['my_test_precisionS']   = 0.0
            dfr['my_moving_precisionB'] = 0.0
            dfr['my_moving_precisionS'] = 0.0
            dfr['y_pred']   = model.predict(X_predict)[0]
            dfr['cat_pred'] = cat_list[cl_list.index(dfr['y_pred'].values[0])]
            dfr['n_pred'] = np.mean(np.array(eval(dfr['cat_pred'].values[0].replace('cat_(', '['))))
            probas = model.predict_proba(X_predict)
            dfr['pS1'] = probas[0][0]
            dfr['pS2'] = probas[0][1]
            dfr['pS3'] = probas[0][2]
            dfr['cS1'] = np.mean(np.array(eval(cat_list[0].replace('cat_(', '['))))
            dfr['cS2'] = np.mean(np.array(eval(cat_list[1].replace('cat_(', '['))))
            dfr['cS3'] = np.mean(np.array(eval(cat_list[2].replace('cat_(', '['))))
            dfr['pB1'] = probas[0][-1]
            dfr['pB2'] = probas[0][-2]
            dfr['pB3'] = probas[0][-3]
            dfr['cB1'] = np.mean(np.array(eval(cat_list[-1].replace('cat_(', '['))))
            dfr['cB2'] = np.mean(np.array(eval(cat_list[-2].replace('cat_(', '['))))
            dfr['cB3'] = np.mean(np.array(eval(cat_list[-3].replace('cat_(', '['))))
            dfr

            # %
            dfr.reset_index().to_sql('predict_clf_rf', engine, if_exists='append', index=False)
            break


# %%
