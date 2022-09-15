# %% 🐭 Excellence
from sys import argv
from rata.utils import parse_argv
import logging as l
li = l.info
l.debug('CMD is: ' + ' '.join(argv))

fake_argv  = 'model_clf_rf.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=9 '
fake_argv += '--X_symbols=EURUSD,NZDUSD '
fake_argv += '--X_include=close,rsi '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=9000 '
fake_argv += '--tstamp=2023-08-31 '
fake_argv += '--test_lenght=800 '
fake_argv += '--nbins=9 '

fake_argv += '--n_estimators=300 '
fake_argv += '--bootstrap=False '
fake_argv += '--class_weight=None '
fake_argv += '--n_jobs=4 '

fake_argv = fake_argv.split()
#argv = fake_argv #### *!
#argv="python3 -u model_clf_rf.py --db_host=192.168.1.83 --symbol=EURUSD --interval=3 --shift=6 --X_symbols=EURUSD --X_include=close,stoch --X_exclude=volatility_kcli --tstamp=2023-01-01T00:00:00 --nrows=7000 --test_lenght=800 --nbins=15 --n_estimators=300 --bootstrap=False --class_weight=None --n_jobs=6 --random_state=36587996".split()

_conf = parse_argv(argv=argv)
#_conf['n_jobs'] = 4
_conf['X_symbols']   = _conf['X_symbols'].split(',')
_conf['X_include']   = _conf['X_include'].split(',')
_conf['X_exclude']   = _conf['X_exclude'].split(',')

if _conf['class_weight'] == 'None':
    _conf['class_weight'] = None

y_target  = _conf['symbol'] + '_' + str(_conf['interval'])
y_target += '_y_close_SROC_' + str(_conf['shift'])
y_target += '_shift-' + str(_conf['shift'])
_conf['y_target'] = y_target
_conf

# %% Global imports
li('Globl imports')
import pandas as pd
import numpy  as np
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
import datetime

#%%
li('Get symbols feateng tables from DB')
engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')

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
    
X_train = X[:-_conf['test_lenght']]
y_train = y[:-_conf['test_lenght']]

X_test = X[-_conf['test_lenght']:]
y_test = y[-_conf['test_lenght']:]

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
model.fit(X_train, y_train)
fit_time = (datetime.datetime.now() - t0).total_seconds()
#%%
dfv  = pd.DataFrame(y_test)
dfv.rename({y_target: 'y_test'}, axis=1, inplace=True)
# common to all models
dfv['y_pred']    = model.predict(X_test)
predict_probas = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
for x in predict_probas.columns:
    dfv['yp_' + x] = predict_probas[x].values

dfv['yp_cl_S']  = dfv['yp_cl_00'] + dfv['yp_cl_01'] + dfv['yp_cl_02']
dfv['yp_cl_B']  = dfv.iloc[:, -3] + dfv.iloc[:, -2] + dfv.iloc[:, -1]

dfr = dict()

dfr['symbol']    = _conf['symbol']
dfr['interval']  = _conf['interval']
dfr['shift']     = _conf['shift']
dfr['X_symbols'] = ','.join(_conf['X_symbols'])
dfr['X_include'] = ','.join(_conf['X_include'])

dfr['test_lenght']   = _conf['test_lenght']
dfr['nrows']         = _conf['nrows']
dfr['n_jobs']        = _conf['n_jobs']
dfr['random_state']  = random_state

dfr['model_tstamp']  = df.index.max()
dfr['model_id']      = str(datetime.datetime.now()).replace(' ', 'T')
dfr['fit_time']      = fit_time

# uncommon to models
dfr['n_estimators']  = _conf['n_estimators']
dfr['nbins']         = _conf['nbins']
dfr['bootstrap']     = _conf['bootstrap']
dfr['class_weight']  = _conf['class_weight']

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

print(cat_list)
print(cl_list)
print(y_train.value_counts())
print(y_test.value_counts())

cm = pd.DataFrame(confusion_matrix(dfv['y_test'], dfv['y_pred'], labels=cl_list))
i = _conf['nbins'] // 3
j = _conf['nbins'] // 5

posS = cm.iloc[:i, :j].sum().sum()
negS = cm.iloc[i:, :j].sum().sum()
my_supportS   = cm.iloc[:j].sum().sum()
my_precisionS = posS / (posS + negS)

posB = cm.iloc[-i:, -j:].sum().sum()
negB = cm.iloc[:-i, -j:].sum().sum()
my_supportB = cm.iloc[-j:].sum().sum()
my_precisionB = posB / (posB + negB)

print(i, j, my_supportB, my_precisionB, my_supportS, my_precisionS)
cm
# %%
dfr['accuracy']      = accuracy_score(dfv['y_test'],  dfv['y_pred'])
dfr['precision']     = precision_score(dfv['y_test'], dfv['y_pred'], average='micro')
dfr['recall']        = recall_score(dfv['y_test'], dfv['y_pred'],    average='micro')
dfr['i']             = i
dfr['j']             = j
dfr['my_precisionB'] = my_precisionB
dfr['my_precisionS'] = my_precisionS
dfr['posB']          = posB
dfr['posS']          = posS
dfr['my_supportB']   = my_supportB
dfr['my_supportS']   = my_supportS
dfr['my_catchB'] = (dfr['posB'] * dfr['my_precisionB']) / dfr['my_supportB']
dfr['my_catchS'] = (dfr['posS'] * dfr['my_precisionS']) / dfr['my_supportS']
dfr['cmd']           = ' '.join(argv) + ' --random_state=' + str(random_state)
dfr = pd.DataFrame([dfr,])

#%%
dfr.reset_index().to_sql('model_clf_rf', engine, if_exists='append', index=False)
if not((my_precisionB > 0.8) or (my_precisionS > 0.8)):
    quit()

# %% COPY
#   ########   ########    ######   #     #
#  #          #        #  #      #   #   #
#  #          #        #  #######     # #
#  #          #        #  #            #
#  #          #        #  #            #
#   ########   ########   #            #  

for reps in range(0,5):
    from sklearn.ensemble import RandomForestClassifier
    random_state = int(datetime.datetime.now().strftime('%S%f'))
    model = RandomForestClassifier( n_estimators=_conf['n_estimators'],
                                    random_state=random_state,
                                    class_weight=_conf['class_weight'],
                                    bootstrap=_conf['bootstrap'],
                                    n_jobs=_conf['n_jobs'])

    t0 = datetime.datetime.now()
    model.fit(X_train, y_train)
    fit_time = (datetime.datetime.now() - t0).total_seconds() 
    #%%
    dfv  = pd.DataFrame(y_test)
    dfv.rename({y_target: 'y_test'}, axis=1, inplace=True)
    # common to all models
    dfv['y_pred']    = model.predict(X_test)
    predict_probas = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
    for x in predict_probas.columns:
        dfv['yp_' + x] = predict_probas[x].values

    dfv['yp_cl_S']  = dfv['yp_cl_00'] + dfv['yp_cl_01'] + dfv['yp_cl_02']
    dfv['yp_cl_B']  = dfv.iloc[:, -3] + dfv.iloc[:, -2] + dfv.iloc[:, -1]

    dfr = dict()

    dfr['symbol']    = _conf['symbol']
    dfr['interval']  = _conf['interval']
    dfr['shift']     = _conf['shift']
    dfr['X_symbols'] = ','.join(_conf['X_symbols'])
    dfr['X_include'] = ','.join(_conf['X_include'])

    dfr['test_lenght']   = _conf['test_lenght']
    dfr['nrows']         = _conf['nrows']
    dfr['n_jobs']        = _conf['n_jobs']
    dfr['random_state']  = random_state

    dfr['model_tstamp']  = df.index.max()
    dfr['model_id']      = str(datetime.datetime.now()).replace(' ', 'T')
    dfr['fit_time']      = fit_time

    # uncommon to models
    dfr['n_estimators']  = _conf['n_estimators']
    dfr['nbins']         = _conf['nbins']
    dfr['bootstrap']     = _conf['bootstrap']
    dfr['class_weight']  = _conf['class_weight']

    #%%
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    print(cat_list)
    print(cl_list)
    print(y_train.value_counts())
    print(y_test.value_counts())

    cm = pd.DataFrame(confusion_matrix(dfv['y_test'], dfv['y_pred'], labels=cl_list))
    i = _conf['nbins'] // 3
    j = _conf['nbins'] // 5

    posS = cm.iloc[:i, :j].sum().sum()
    negS = cm.iloc[i:, :j].sum().sum()
    my_supportS   = cm.iloc[:j].sum().sum()
    my_precisionS = posS / (posS + negS)

    posB = cm.iloc[-i:, -j:].sum().sum()
    negB = cm.iloc[:-i, -j:].sum().sum()
    my_supportB = cm.iloc[-j:].sum().sum()
    my_precisionB = posB / (posB + negB)

    print(i, j, my_supportB, my_precisionB, my_supportS, my_precisionS)
    cm
    # %%
    dfr['accuracy']      = accuracy_score(dfv['y_test'],  dfv['y_pred'])
    dfr['precision']     = precision_score(dfv['y_test'], dfv['y_pred'], average='micro')
    dfr['recall']        = recall_score(dfv['y_test'], dfv['y_pred'],    average='micro')
    dfr['i']             = i
    dfr['j']             = j
    dfr['my_precisionB'] = my_precisionB
    dfr['my_precisionS'] = my_precisionS
    dfr['posB']          = posB
    dfr['posS']          = posS
    dfr['my_supportB']   = my_supportB
    dfr['my_supportS']   = my_supportS
    dfr['my_catchB'] = (dfr['posB'] * dfr['my_precisionB']) / dfr['my_supportB']
    dfr['my_catchS'] = (dfr['posS'] * dfr['my_precisionS']) / dfr['my_supportS']
    dfr['cmd']           = ' '.join(argv) + ' --random_state=' + str(random_state)
    dfr = pd.DataFrame([dfr,])
    #%%
    dfr.reset_index().to_sql('model_clf_rf', engine, if_exists='append', index=False)