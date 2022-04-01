# %%
import pandas as pd
pd.options.display.max_rows = 50000
pd.options.display.width = 200
df = pd.read_csv('/home/selknam/var/BTCUSD_3.forecast_1.csv')

columns = ['tstamp', 'tstamp_forecast']
for c in df.columns:
    if 'proba_1' in c:
        #c_rename = c.replace('Y_BTCUSD_3_close_ROC_', '').replace('_shift', '')
        c_rename = c.replace('Y_BTCUSD_3_close_ROC_', '').replace('_shift', '').replace('_proba_1', '')
        c_rename = c_rename.replace('3_3_', '3').replace('6_6_', '6').replace('9_9_', '9')

        columns.append(c_rename)
        df.rename({c: c_rename}, axis=1, inplace=True)

df = df[columns]
df['tstamp_forecast'] = df['tstamp_forecast'].str.slice(5, 19)
df['tstamp'] = pd.to_datetime(df['tstamp'])
df.set_index('tstamp', inplace=True)
df.sort_values(by='tstamp', inplace=True)
df = df[['tstamp_forecast', '3B', '6B', '9B', '3S', '6S', '9S']]
#print(df.tail(5000))

df[ (
    (df['3S'] > 0.01) &
    (df['6S'] > 0.20) &
    (df['9S'] > 0.30)
    ) | (
    (df['3B'] > 0.01) &
    (df['6B'] > 0.20) &
    (df['9B'] > 0.30)
    )
]

# %%
import pandas as pd
pd.options.display.max_rows = 50000
pd.options.display.width = 200
df = pd.read_csv('/home/selknam/var/BTCUSD_3.forecast_2.csv')

columns = ['tstamp', 'tstamp_forecast']
for c in df.columns:
    if 'proba_1' in c:
        c_rename = c.replace('Y_BTCUSD_3_close_ROC_', '').replace('_shift', '').replace('_proba_1_', '_')
        c_rename = c_rename.replace('3_3_', '3').replace('6_6_', '6').replace('9_9_', '9')
        columns.append(c_rename)
        df.rename({c: c_rename}, axis=1, inplace=True)

df = df[columns]
df['tstamp_forecast'] = df['tstamp_forecast'].str.slice(5, 19)
df['tstamp'] = pd.to_datetime(df['tstamp'])
df.set_index('tstamp', inplace=True)
df.sort_values(by='tstamp', inplace=True)
#print(df.tail(5000))

df_3B_r1 = df[['tstamp_forecast', '3B_r1']].copy()
df_3B_r1.rename({'3B_r1': '3B'}, axis=1, inplace=True)
df_3B_r2 = df[['tstamp_forecast', '3B_r2']].copy()
df_3B_r2.rename({'3B_r2': '3B'}, axis=1, inplace=True)
df_3B_r3 = df[['tstamp_forecast', '3B_r3']].copy()
df_3B_r3.rename({'3B_r3': '3B'}, axis=1, inplace=True)

df_3S_r1 = df[['tstamp_forecast', '3S_r1']].copy()
df_3S_r1.rename({'3S_r1': '3S'}, axis=1, inplace=True)
df_3S_r2 = df[['tstamp_forecast', '3S_r2']].copy()
df_3S_r2.rename({'3S_r2': '3S'}, axis=1, inplace=True)
df_3S_r3 = df[['tstamp_forecast', '3S_r3']].copy()
df_3S_r3.rename({'3S_r3': '3S'}, axis=1, inplace=True)

df_6B_r1 = df[['tstamp_forecast', '6B_r1']].copy()
df_6B_r1.rename({'6B_r1': '6B'}, axis=1, inplace=True)
df_6B_r2 = df[['tstamp_forecast', '6B_r2']].copy()
df_6B_r2.rename({'6B_r2': '6B'}, axis=1, inplace=True)
df_6B_r3 = df[['tstamp_forecast', '6B_r3']].copy()
df_6B_r3.rename({'6B_r3': '6B'}, axis=1, inplace=True)

df_6S_r1 = df[['tstamp_forecast', '6S_r1']].copy()
df_6S_r1.rename({'6S_r1': '6S'}, axis=1, inplace=True)
df_6S_r2 = df[['tstamp_forecast', '6S_r2']].copy()
df_6S_r2.rename({'6S_r2': '6S'}, axis=1, inplace=True)
df_6S_r3 = df[['tstamp_forecast', '6S_r3']].copy()
df_6S_r3.rename({'6S_r3': '6S'}, axis=1, inplace=True)

df_9B_r1 = df[['tstamp_forecast', '9B_r1']].copy()
df_9B_r1.rename({'9B_r1': '9B'}, axis=1, inplace=True)
df_9B_r2 = df[['tstamp_forecast', '9B_r2']].copy()
df_9B_r2.rename({'9B_r2': '9B'}, axis=1, inplace=True)
df_9B_r3 = df[['tstamp_forecast', '9B_r3']].copy()
df_9B_r3.rename({'9B_r3': '9B'}, axis=1, inplace=True)

df_9S_r1 = df[['tstamp_forecast', '9S_r1']].copy()
df_9S_r1.rename({'9S_r1': '9S'}, axis=1, inplace=True)
df_9S_r2 = df[['tstamp_forecast', '9S_r2']].copy()
df_9S_r2.rename({'9S_r2': '9S'}, axis=1, inplace=True)
df_9S_r3 = df[['tstamp_forecast', '9S_r3']].copy()
df_9S_r3.rename({'9S_r3': '9S'}, axis=1, inplace=True)

df_3B = pd.concat([df_3B_r1, df_3B_r2, df_3B_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])
df_3S = pd.concat([df_3S_r1, df_3S_r2, df_3S_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])

df_6B = pd.concat([df_6B_r1, df_6B_r2, df_6B_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])
df_6S = pd.concat([df_6S_r1, df_6S_r2, df_6S_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])

df_9B = pd.concat([df_9B_r1, df_9B_r2, df_9B_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])
df_9S = pd.concat([df_9S_r1, df_9S_r2, df_9S_r3]).reset_index().set_index(['tstamp', 'tstamp_forecast'])
df_2 = pd.concat([df_3B, df_6B, df_9B, df_3S, df_6S, df_9S], axis=1).sort_index()
df_2
#
# %%
