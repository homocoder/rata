# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv = 'feateng.py --db_host=localhost --symbol=BTCUSD --interval=3 '
fake_argv = fake_argv.split()
argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt

def join_indicators_tv(df_indicators, df_tv):
    # Join df indicators and df tv strategies
    df_indicators.sort_index(inplace=True)
    first_tstamp_indicators = df_indicators.index[0]
    last_tstamp_indicators  = df_indicators.index[-1]

    df_tv.sort_index(inplace=True)
    df_tv.drop('tstamp', axis=1, inplace=True)
    first_tstamp_tv = df_tv.index[0]
    last_tstamp_tv  = df_tv.index[-1]

    df_tv.drop(df_tv.tail(1).index, inplace=True) # drop last 2 rows
    df_tv = df_tv[(df_tv.index >= first_tstamp_indicators) & (df_tv.index <= last_tstamp_indicators)]
    df_indicators = df_indicators[(df_indicators.index >= first_tstamp_tv) & (df_indicators.index <= last_tstamp_tv)]

    df_feateng = df_indicators.join(df_tv).fillna(axis='rows', method='ffill').sort_index()
    df_feateng.dropna(inplace=True)
    df_feateng['tstamp'] = df_feateng.index
    return df_feateng

def custom_resample_open(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike.iloc[0]

def custom_resample_close(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike.iloc[-1]

def check_time_gaps(df):
    print('\n##### Checking time gaps for: ', collection, 'To interval:', interval, ' #####')
    df_diff_intervals = pd.DataFrame(df['tstamp'])
    df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
    df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
    df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

    df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 1]
    print('Len: ', len(df_delta_minutes))
    if len(df_delta_minutes > 0):
        print('First: ', df.iloc[0]['tstamp'])
        print('Last: ',  df.iloc[-1]['tstamp'])
    print('Gaps are in minutes. Showing gaps interval*3')
    print(df_delta_minutes)

def read_tv_strategy(file_name):
    from unicodedata import normalize
    from pandas import read_csv
    
    df = read_csv(file_name)

    columns = list()
    for text in df.columns:
        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3
            pass
        for c in ['#', '/', '$', '%', ' ', '.', '-']:
            text = text.replace(c, '')
        text = normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8").lower()
        columns.append(text)

    strategy_name = file_name.split('_')[-4]

    df.columns = columns
    df = df[['datetime', 'trade', 'signal', 'type', 'price', 'profit']].reset_index(drop=True)
    df1 = df[df['type'].str.contains('Entry')].set_index('trade')
    df2 = df[df['type'].str.contains('Exit')].set_index('trade')
    df = df1.join(df2, rsuffix='_exit').reset_index()
    datetime_tmp = pd.to_datetime(df['datetime']).dt.round(freq='3min')

    for c in df.columns:
        df.rename({c: c + '_' + strategy_name}, axis=1, inplace=True)
    df['tstamp'] = datetime_tmp

    df.set_index(df['tstamp'], inplace=True)
    return df

# %%
from pymongo import MongoClient

client = MongoClient(_conf['db_host'], 27017)
db = client['rates']
list_collection_names = db.list_collection_names()
list_collection_names.sort()

collection = _conf['symbol'] + '_1'
mydoc = db[collection].find({})
df = pd.DataFrame(mydoc)
df.sort_values(by='tstamp', ascending=True, inplace=True)
symbol    = df[['symbol'  ]].iloc[0]['symbol']
interval  = int(df[['interval']].iloc[0]['interval']) # always 1

# %%

if interval != _conf['interval']:
    print('\n##### Resampling: ', collection, ' #####')
    interval = _conf['interval']
    print('To interval:', interval)
    resample_rule = str(_conf['interval']) + 'min'  
    ts_open   = df[['tstamp', 'open'  ]].resample(resample_rule, on='tstamp').apply(custom_resample_open)['open']
    ts_high   = df[['tstamp', 'high'  ]].resample(resample_rule, on='tstamp').max()['high']
    ts_low    = df[['tstamp', 'low'   ]].resample(resample_rule, on='tstamp').min()['low']
    ts_close  = df[['tstamp', 'close' ]].resample(resample_rule, on='tstamp').apply(custom_resample_close)['close']
    ts_volume = df[['tstamp', 'volume']].resample(resample_rule, on='tstamp').sum()['volume']

    df_resample = pd.concat([ts_open, ts_high, ts_low, ts_close, ts_volume], axis=1).sort_index()
    df_resample['symbol']   = symbol
    df_resample['interval'] = interval
    del df
    df = df_resample.copy()
    df.reset_index(drop=False, inplace=True)
    df.dropna(inplace=True)
else:
    print('\n##### Not resampling: ', collection, ' #####')

check_time_gaps(df)
# %% 🐭
# Technical Indicators
# TODO: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
import ta
MACD = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=10, window_slow=21, window_sign=9)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=7, window_slow=18, window_sign=7)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

MACD = ta.trend.MACD(close=df['close'], window_fast=8, window_slow=21, window_sign=8)
macd        = pd.DataFrame(MACD.macd())
macd_diff   = pd.DataFrame(MACD.macd_diff())
macd_signal = pd.DataFrame(MACD.macd_signal())
df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

KST      = ta.trend.KSTIndicator(close=df['close'])
kst_sig  = pd.DataFrame(KST.kst_sig())
kst      = pd.DataFrame(KST.kst())
kst_diff = pd.DataFrame(KST.kst_diff())
df = pd.concat([df, kst, kst_sig, kst_diff], axis=1)

ROC  = ta.momentum.roc(close=df['close'], window=9)
RSI  = ta.momentum.rsi(close=df['close'])
KAMA = ta.momentum.kama(close=df['close'])
OBV  = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])

ROC_3 = ta.momentum.ROCIndicator(close=df['close'], window=3)
roc_3 = pd.DataFrame(ROC_3.roc().rename('roc_3'))
df = pd.concat([df, roc_3], axis=1)

ROC_6 = ta.momentum.ROCIndicator(close=df['close'], window=6)
roc_6 = pd.DataFrame(ROC_6.roc().rename('roc_6'))
df = pd.concat([df, roc_6], axis=1)

ROC_9 = ta.momentum.ROCIndicator(close=df['close'], window=9)
roc_9 = pd.DataFrame(ROC_9.roc().rename('roc_9'))
df = pd.concat([df, roc_9], axis=1)

df = pd.concat([df, RSI, KAMA, OBV], axis=1)

df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

for c in df.columns.drop(['tstamp', 'close', 'symbol', 'interval']):
    for i in range(3, 10, 3):
        df[str(c) + '_ROCs_' + str(i)] = df[c].pct_change(i) * 100

df.set_index(df['tstamp'], inplace=True)
df_feateng = df.copy()
del df
#%%
file_list = [
    '../tvdata/BTCUSD_3m_Momentum_Strategy_2022-03-15_b67ff.csv',
    '../tvdata/BTCUSD_3m_Pivot_Reversal_Strategy_2022-03-15_433a0.csv',
    '../tvdata/BTCUSD_3m_Parabolic_SAR_Strategy_2022-03-15_409cf.csv'
]
for file_name in file_list:
    pass
    #df_tv = read_tv_strategy(file_name=file_name)
    #print(file_name)
    #print(df_tv[(df_tv.index.minute % 3) > 0].index)
    #df_feateng = join_indicators_tv(df_indicators=df_feateng, df_tv=df_tv)
    #check_time_gaps(df_feateng)
# %%
X_prefix = 'X_' + symbol + '_' + str(interval) + '_'
for c in df_feateng.columns:
    df_feateng.rename({c: X_prefix + c}, axis=1, inplace=True)

Y_prefix = 'Y_' + symbol + '_' + str(interval) + '_'

# Y for regression
df_feateng[Y_prefix + 'roc_3_shift_4']  = df_feateng[X_prefix + 'roc_3'].shift(-4) # to see the future
df_feateng[Y_prefix + 'roc_6_shift_7']  = df_feateng[X_prefix + 'roc_6'].shift(-7) # to see the future
df_feateng[Y_prefix + 'roc_9_shift_10'] = df_feateng[X_prefix + 'roc_9'].shift(-10) # to see the future

# Y for classification
df_feateng[Y_prefix + 'roc_3_shift_4_B'] = 0
df_feateng[Y_prefix + 'roc_3_shift_4_B'] = df_feateng[Y_prefix + 'roc_3_shift_4_B'].mask(df_feateng[Y_prefix + 'roc_3_shift_4'] > 0.3, 1)

df_feateng[Y_prefix + 'roc_6_shift_7_B'] = 0
df_feateng[Y_prefix + 'roc_6_shift_7_B'] = df_feateng[Y_prefix + 'roc_6_shift_7_B'].mask(df_feateng[Y_prefix + 'roc_6_shift_7'] > 0.3, 1)

df_feateng[Y_prefix + 'roc_9_shift_10_B'] = 0
df_feateng[Y_prefix + 'roc_9_shift_10_B'] = df_feateng[Y_prefix + 'roc_9_shift_10_B'].mask(df_feateng[Y_prefix + 'roc_9_shift_10'] > 0.3, 1)


df_feateng[Y_prefix + 'roc_3_shift_4_S'] = 0
df_feateng[Y_prefix + 'roc_3_shift_4_S'] = df_feateng[Y_prefix + 'roc_3_shift_4_S'].mask(df_feateng[Y_prefix + 'roc_3_shift_4'] < -0.3, 1)

df_feateng[Y_prefix + 'roc_6_shift_7_S'] = 0
df_feateng[Y_prefix + 'roc_6_shift_7_S'] = df_feateng[Y_prefix + 'roc_6_shift_7_S'].mask(df_feateng[Y_prefix + 'roc_6_shift_7'] < -0.3, 1)

df_feateng[Y_prefix + 'roc_9_shift_10_S'] = 0
df_feateng[Y_prefix + 'roc_9_shift_10_S'] = df_feateng[Y_prefix + 'roc_9_shift_10_S'].mask(df_feateng[Y_prefix + 'roc_9_shift_10'] < -0.3, 1)

#df_feateng.dropna(inplace=True)
df_feateng['tstamp'] = df_feateng.index
check_time_gaps(df_feateng)

df_feateng.to_csv('BTCUSD_3m.feateng.csv')
# END OF FEATENG

# %%
# START OF FEATSEL
featsel = ['kst_diff',
'momentum_pvo',
'momentum_pvo_hist',
'momentum_pvo_signal',
'momentum_tsi',
'momentum_uo',
'trend_adx',
'trend_adx_neg',
'trend_adx_pos',
'trend_aroon_ind',
'trend_dpo',
'trend_ichimoku_b',
'trend_kst_sig',
'trend_mass_index',
'trend_psar_down',
'trend_psar_up',
'trend_sma_slow',
'trend_stc',
'trend_visual_ichimoku_a',
'trend_visual_ichimoku_b',
'volatility_atr',
'volatility_bbw',
'volatility_dcw',
'volatility_kcw',
'volatility_ui',
'volume_adi',
'volume_cmf',
'volume_nvi',
'volume_obv',
'volume_sma_em',

'MACD',
'kama',
'close']

features_selected = set()
for i in featsel:
    for j in df_feateng.columns:
        if (i in j) and ('Y_' not in j):
            features_selected.add(j)
features_selected = list(features_selected)

df_feateng = df_feateng[features_selected]

featsel = ['ROCs']
features_selected = set()
for i in featsel:
    for j in df_feateng.columns:
        if (i in j) and ('Y_' not in j):
            features_selected.add(j)
features_selected = list(features_selected)

X = df_feateng[features_selected]


# %%
y = df_feateng[Y_prefix + 'roc_9_shift_10_S']
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, random_state=int(dt.datetime.now().strftime('%S%f')), shuffle=True)
# create model
clf = RandomForestClassifier(random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1)
clf.class_weight = "balanced"
# evaluate model
scores = cross_val_score(clf, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores
# %%
from sklearn.metrics import precision_score
df_cv = X
c = 1
for train_index, test_index in cv.split(X, y):
    # create model
    del clf
    clf = RandomForestClassifier(random_state=int(dt.datetime.now().strftime('%S%f')), n_jobs=-1)
    clf.class_weight = "balanced"
    print('Fold ', c, len(train_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    X_test['fold_' + str(c) + '_y_pred'] = y_pred
    y_pred = X_test['fold_' + str(c) + '_y_pred']
    y_proba = pd.DataFrame(y_proba, columns=['fold_' + str(c) + '_y_proba_0', 'fold_' + str(c) + '_y_proba_1'])
    y_proba.set_index(X_test.index, inplace=True)
    df_cv = df_cv.join(y_proba)
    print(precision_score(y_test, y_pred, pos_label=1, average='binary'))
    c = c + 1

# %%
df_cv['y_proba_1'] = df_cv[['fold_1_y_proba_1', 'fold_2_y_proba_1', 'fold_3_y_proba_1', 'fold_4_y_proba_1', 'fold_5_y_proba_1']].sum(axis=1)

df_cv.to_csv(Y_prefix + 'roc_9_shift_10_S.csv')
# %%
