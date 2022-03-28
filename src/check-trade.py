# %%
import pandas as pd
pd.options.display.max_rows = 50000
pd.options.display.width = 200
df = pd.read_csv('/home/selknam/var/BTCUSD_3.forecast_1.csv')

columns = ['tstamp']
for c in df.columns:
    if 'proba_1' in c:
        c_rename = c.replace('Y_BTCUSD_3_close_ROC_', '').replace('_shift', '')
        columns.append(c_rename)
        df.rename({c: c_rename}, axis=1, inplace=True)

df = df[columns]
df['tstamp'] = pd.to_datetime(df['tstamp'])
df.set_index('tstamp', inplace=True)
df.sort_values(by='tstamp', inplace=True)
print(df)


# %%
