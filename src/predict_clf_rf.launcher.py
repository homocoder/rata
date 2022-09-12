# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'model_clf_rf.py --db_host=192.168.1.83 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=9 '

fake_argv = fake_argv.split()
argv = fake_argv #### *!
#argv="python3 -u model_clf_rf.py --db_host=192.168.1.83 --symbol=EURUSD --interval=1 --shift=15 --X_symbols=EURUSD,AUDUSD --X_include=rsi,* --X_exclude=volatility_kcli --tstamp=2022-08-10T00:00:00 --nrows=5000 --test_lenght=800 --nbins=24 --n_estimators=300 --bootstrap=True --class_weight=None".split()
_conf = parse_argv(argv=argv)

_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'

_conf

# %% Global imports
import pandas as pd
import numpy  as np
from rata.ratalib import check_time_gaps
from sqlalchemy import create_engine
import datetime

# %%
engine = create_engine(_conf['url'])
sql  = """
select model_tstamp, symbol, interval, shift, nbins, 
  round("my_precisionB"::numeric, 2) as "my_precisionB", round("my_precisionS"::numeric, 2) as "my_precisionS",
  "posB", "posS", "my_supportB", "my_supportS", cmd 
from model_clf_rf
where
  (("my_precisionB" >= 0.85) or ("my_precisionS" >= 0.85)) and
  model_tstamp >= (now() - interval '240 hours' )
"""

with engine.connect() as conn:
    dfa = pd.read_sql_query(sql, conn)

# %%
df = pd.DataFrame()
interval     = [1, 3]
shift        = [6, 9, 15, 30, 60, 90]
my_precision = ['my_precisionS', 'my_precisionB']

for i in interval:
    for s in shift:
        for m in my_precision:
            ##
            my_thresh = 1.00
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))
            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue
            ##
            my_thresh = 1.00
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue
            ##
            my_thresh = 1.00
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.95
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.95
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.95
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.90
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.90
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.90
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.85
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.85
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.85
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.85
            hours     = 12
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue

            ##
            my_thresh = 0.85
            hours     = 24
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            print(i, s, m, my_thresh, hours, len(dfq))

            if len(dfq) >= 3:
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                continue
# %%
