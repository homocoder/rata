# %%
import pandas as pd
from unicodedata import normalize
file_name = '../tvdata/BTCUSD_1m_Momentum_Strategy_2022-03-15_63b7a.csv'
df = pd.read_csv(file_name)

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

df.columns = columns
df = df[['datetime', 'trade', 'signal', 'type', 'price', 'profit']].reset_index(drop=True)

#%%
df1 = df[df['type'].str.contains('Entry')].set_index('trade')
df2 = df[df['type'].str.contains('Exit')].set_index('trade')
df = df1.join(df2, rsuffix='_exit').reset_index()



#%%
