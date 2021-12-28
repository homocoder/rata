# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'dbmaintain --db_host=localhost --db_port=27017 --dbs_prefix=rata'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf

# %%
# Global imports
import pandas as pd
import datetime as dt

# %%
## */* RATES maintenance */* ##
from pymongo import MongoClient
from rata.marketon import get_data

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_rates']

for db_col in db.list_collection_names():

    collection = db[db_col]
    mydoc = collection.find({})
    df = pd.DataFrame(mydoc)
    if len(df) == 0:
        print('No data in ', db_col)
    else:
        print('Cleaning ', db_col)
        #df['status'] = 'ok'
        df = df.groupby(['close', 'high', 'interval', 'low', 'open', 'status', 'symbol', 'tstamp', 'unix_tstamp', 'volume'], as_index=False).max()
        df = df[['query_tstamp', 'unix_tstamp', 'tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].sort_values('tstamp')
        
        collection.drop()
        db_bkp = client[_conf['dbs_prefix'] + '_rates']
        #db_bkp = client['rata' + '_rates']
        collection = db_bkp[db_col]
        collection.insert_many(df.to_dict(orient='records'))

client.close()
# %%

# %%
#def generate_launchers(txt, txt_replace, interval, t1, t2):
from datetime import datetime
from sklearn.utils import shuffle

txt = """interval=5
sleep 1
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecasts_binclf_launcher.py \
    --db_conf=conf/db.json \
    --symbol_conf=conf/rates_launcher.$interval.json \
    --forecasts_binclf_conf=conf/forecasts_binclf.json \
    --forecast_datetime=0000-00-00T00:00:00 
"""
txt_replace = "0000-00-00T00:00:00"

t1 = datetime(2021, 12, 26, 22, 00, 0)
t2 = datetime(2021, 12, 28, 14, 00, 0)

import pandas as pd
df = pd.DataFrame(pd.date_range(start=t1, end=t2, freq='5min'))
df = shuffle(df)
cmd = ''
for i in range(0, len(df)):
    cmd += txt.replace(txt_replace, str(df.iloc[i][0]).replace(' ', 'T'))

print(cmd)
# %%
