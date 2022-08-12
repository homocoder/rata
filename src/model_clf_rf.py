# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'model_clf_rf.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=3 --shift=3 '
fake_argv += '--X_symbols=EURUSD '
fake_argv += '--X_include=vpt,rsi,stoch,others_cr,macd,kst,adx,cci,dch,open,high,low,close,volume,obv '
fake_argv += '--X_exclude=volatility_kcli '

fake_argv += '--nrows=9000 ' 
fake_argv += '--tstamp=2022-07-28 ' 
fake_argv += '--test_lenght=1800 '
fake_argv += '--nbins=14 '

fake_argv += '--iterations=30 '
fake_argv += '--learning_rate=0.9 '
fake_argv += '--depth=6 '
fake_argv += '--l2_leaf_reg=3 '
fake_argv += '--loss_function=MAE '

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

#%%
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
#check_time_gaps(df, {'symbol': s, 'interval': 3})
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
    for exc in _conf['X_exclude']:
        if exc in c:
            ys_todelete.append(c)

df = df[:-9]
#columns containing NaNs
dfnans = pd.DataFrame(df.isnull().sum())
nancols = list(dfnans[dfnans[0] > 0].index)
ys_todelete = ys_todelete + nancols

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

model = RandomForestClassifier(n_estimators=300, random_state=int(datetime.datetime.now().strftime('%S%f')),
                                n_jobs=-1)

t0 = datetime.datetime.now()
model.fit(X_train, y_train)
fit_time = int((datetime.datetime.now() - t0).total_seconds() / 60)
#%%
dfv  = pd.DataFrame(y_test)
dfv.rename({y_target: 'y_test'}, axis=1, inplace=True)
# common to all models
dfv['y_pred']    = model.predict(X_test)
predict_probas = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
for x in predict_probas.columns:
    dfv['yp_' + x] = predict_probas[x].values

dfv['yp_cl_B']  = dfv.iloc[:, -3] + dfv.iloc[:, -2] + dfv.iloc[:, -1]
dfv['yp_cl_S']  = dfv['yp_cl_00'] + dfv['yp_cl_01'] + dfv['yp_cl_02']

dfv['symbol']    = _conf['symbol']
dfv['interval']  = _conf['interval']
dfv['shift']     = _conf['shift']
dfv['X_symbols'] = ','.join(_conf['X_symbols'])

dfv['test_lenght']   = _conf['test_lenght']
dfv['nrows']         = _conf['nrows']

dfv['model_tstamp']  = df.index.max()
dfv['model_id']      = str(datetime.datetime.now()).replace(' ', 'T')
dfv['fit_time']      = fit_time

# uncommon to models
dfv['iterations']    = _conf['iterations']
dfv['learning_rate'] = _conf['learning_rate']
dfv['depth']         = _conf['depth']
dfv['l2_leaf_reg']   = _conf['l2_leaf_reg']

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
print(cat_list)
print(cl_list)
print(y_train.value_counts())
print(y_test.value_counts())
print(accuracy_score(dfv['y_test'], dfv['y_pred']))
print(precision_score(dfv['y_test'], dfv['y_pred'], average='macro'))
print(recall_score(dfv['y_test'], dfv['y_pred'], average='macro'))
cm = pd.DataFrame(confusion_matrix(dfv['y_test'], dfv['y_pred'], labels=cl_list))
i = _conf['nbins'] // 3
j = _conf['nbins'] // 5
posB = cm.iloc[-i:, -j:].sum().sum()
negB = cm.iloc[:-i, -j:].sum().sum()
my_presicionB = posB / (posB + negB)

posS = cm.iloc[:i, :j].sum().sum()
negS = cm.iloc[i:, :j].sum().sum()
my_presicionS = posS / (posS + negS)

print(my_presicionB, my_presicionS)
cm
#%%
# second RFC
map_BS = {
    'cl_00': 'S',
    'cl_01': 'S',
    'cl_02': 'S',
    'cl_03': '0',
    'cl_04': '0',
    'cl_05': '0',
    'cl_06': '0',
    'cl_07': '0',
    'cl_08': '0',
    'cl_09': '0',
    'cl_10': '0',
    'cl_11': 'B',
    'cl_12': 'B',
    'cl_13': 'B'
}
cl_list = ['S', '0', 'B']
X2 = dfv.iloc[:, 2: _conf['nbins'] + 4]
y2 = dfv['y_test']
y2 = y2.map(map_BS)
X_train = X2[:-_conf['test_lenght'] // 2]
y_train = y2[:-_conf['test_lenght'] // 2]

X_test = X2[-_conf['test_lenght'] // 2:]
y_test = y2[-_conf['test_lenght'] // 2:]

model = RandomForestClassifier(n_estimators=300, random_state=int(datetime.datetime.now().strftime('%S%f')),
                                n_jobs=-1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=cl_list))

i = _conf['nbins'] // 3
j = _conf['nbins'] // 5
print(i, j)
posB = cm.iloc[-i:, -j:].sum().sum()
negB = cm.iloc[:-i, -j:].sum().sum()
my_presicionB = posB / (posB + negB)

posS = cm.iloc[:i, :j].sum().sum()
negS = cm.iloc[i:, :j].sum().sum()
my_presicionS = posS / (posS + negS)

print(my_presicionB, my_presicionS)
cm
#%%
dffi = pd.DataFrame(model.feature_names_)
dffi.rename({0: 'feature_name'}, axis=1, inplace=True)
dffi['feature_importance'] = model.feature_importances_
dffi.sort_values('feature_importance', ascending=False, inplace=True)
dffi.head(45)

# %%

#%%
#engine = create_engine('postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata')
#dfv.reset_index().to_sql('model_catboost', engine, if_exists='append', index=False)
 # %%



# %%

