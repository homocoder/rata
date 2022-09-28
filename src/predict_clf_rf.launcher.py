# %% 🐭
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'predict_clf_rf.launcher.py --db_host=192.168.1.84 '
fake_argv += '--symbol=EURUSD --interval=1 --shift=15 '
fake_argv += '--my_precision=my_precisionS,my_precisionB '

fake_argv = fake_argv.split()
#argv = fake_argv #### *!
#argv="python3 -u predict_clf_rf.py --db_host=192.168.1.84 --symbol=EURUSD --interval=1 --shift=15 --X_symbols=EURUSD,AUDUSD --X_include=rsi,* --X_exclude=volatility_kcli --tstamp=2022-08-10T00:00:00 --nrows=5000 --test_lenght=800 --nbins=24 --n_estimators=300 --bootstrap=True --class_weight=None".split()

_conf = parse_argv(argv=argv)

if type(_conf['interval']) is int:
    _conf['interval'] = [_conf['interval'],]
else:
    _conf['interval'] = [int(i) for i in _conf['interval'].split(',')]

if type(_conf['shift']) is int:
    _conf['shift'] = [_conf['shift'],]
else:
    _conf['shift']        = [int(i) for i in _conf['shift'].split(',')]

_conf['my_precision'] = _conf['my_precision'].split(',')

_conf['url'] = 'postgresql+psycopg2://rata:acaB.1312@' + _conf['db_host'] + ':5432/rata'

_conf
# %% Global imports
import pandas as pd
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
interval     = _conf['interval']
shift        = _conf['shift']
my_precision = _conf['my_precision']

for i in interval:
    for s in shift:
        for m in my_precision:
            ##
            my_thresh = 1.00
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 1.00
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 1.00
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.95
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.95
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.95
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.90
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.90
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.90
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.85
            hours     = 3
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.85
            hours     = 6
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.85
            hours     = 9
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.85
            hours     = 12
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue
            ##
            my_thresh = 0.85
            hours     = 24
            dfq = dfa[(dfa['interval'] == i) & (dfa['shift'] == s) & (dfa[m] >= my_thresh) & (dfa['model_tstamp']  >= (datetime.datetime.now()-datetime.timedelta(hours=hours)))]
            
            if len(dfq) >= 3:
                dfq['my_precision'] = m
                df = pd.concat([df, dfq.sample(3)]) # take rand 3
                print(i, s, m, my_thresh, hours, len(dfq))
                continue

# %%
df['cmd'] = df['cmd'].str.replace('model_clf_rf.py', 'python3 -u predict_clf_rf.py')
df['cmd'] = df['cmd'].str.cat(df['my_precision'], sep=' --my_precision=')
df['cmd'] = df['cmd'].str.cat(df['my_precisionB'].astype('str'), sep=' --my_test_precisionB=')
df['cmd'] = df['cmd'].str.cat(df['my_precisionS'].astype('str'), sep=' --my_test_precisionS=')
df['cmd'] = df['cmd'] + ' & \n'

# %%
from subprocess import getoutput
interval     = _conf['interval']
shift        = _conf['shift']
my_precision = _conf['my_precision']

for i in interval:
    for s in shift:
        for m in my_precision:
            print(i, s, m)
            cmd = 'ps -fea | grep predict_clf_rf.py | grep "\-\-interval=' + str(i) + ' " | grep "\-\-shift=' + str(s) + ' " | grep "\-\-my_precision=' + str(m) + ' "'
            out = getoutput(cmd)
            if len(out) > 0:
                for o in out.split('\n'):
                    pid = o.split()[1]
                    print(pid)
                    cmd = 'kill -15 ' + pid
                    print(cmd, getoutput(cmd)) # KILL

            filename = '/home/selknam/var/scripts/predict_clf_rf_launcher.i-' + str(i) + '.s-' + str(s) + '.m-' + m + '.bash'
            with open(filename, 'wt') as fd:
                fd.writelines(list(df[(df['interval'] == i) & (df['shift'] == s) & (df['my_precision'] == m)]['cmd'].values))
            print(getoutput('bash ' + filename))
# %%
