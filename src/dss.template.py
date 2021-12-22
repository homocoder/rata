# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from pymongo import MongoClient

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_conf = {'db_host': 'localhost',
         'db_port': 27017,
         'dbs_prefix': 'rt'
}

client = MongoClient(_conf['db_host'], _conf['db_port'])
db = client[_conf['dbs_prefix'] + '_forecasts_binclf']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_out= pd.DataFrame()

for collection in db.list_collection_names():
    mydoc = db[collection].find({}).sort('tstamp', 1)
    df = pd.DataFrame(mydoc)
    df_out = df_out.append(df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
rt_forecasts_binclf = dataiku.Dataset("rt_forecasts_binclf")
rt_forecasts_binclf.write_with_schema(df_out)
