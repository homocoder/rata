# %%
import pandas as pd
pd.date_range(start='2022-09-11T21:15:00', periods=1, freq='3h')

# %%
from random import shuffle
import pandas as pd

model_params = {
    'symbol'        : ['EURUSD'],
    'interval'      : [1, 3],
    'shift'         : [6, 6, 9, 9, 15, 15, 30, 60, 90],
    'X_symbols'     : ['EURUSD', 'EURUSD,GBPUSD', 'EURUSD,AUDUSD', 'EURUSD,NZDUSD', 'EURUSD,USDCAD', 'EURUSD,USDJPY', 'EURUSD,USDCHF'],
    'X_include'     : ['close,atr', 'close,obv', 'close,vpt', 'close,rsi', 'close,stoch', 'close,others_cr', 'close,macd', 'close,kst', 'close,adx', 'close,cci', 'close,dch', 'open,low,high,close,volume,obv,atr,macd,rsi', 'atr,vpt,rsi,stoch,others_cr,macd,kst,adx,cci,dch,open,high,low,close,volume,obv'],
    'X_exclude'     : ['volatility_kcli'],
    'tstamp'        : pd.date_range(start='2023-01-01T00:00:01', periods=1, freq='3h'),
    'nrows'         : [7000],
    'test_lenght'   : [800],
    'nbins'         : [9, 12, 15],
    'n_estimators'  : [300],
    'bootstrap'     : ['True', 'False'],
    'class_weight'  : ['None', 'balanced', 'balanced_subsample'],
    'n_jobs'        : [4]
}

for i in model_params:
    print(i)
    for j in model_params[i]:
        print(i, j)

launcher_cmds = list()
for symbol in model_params['symbol']:
    for interval in model_params['interval']:
        for shift in model_params['shift']:
            for X_symbols in model_params['X_symbols']:
                for X_include in model_params['X_include']:
                    for X_exclude in model_params['X_exclude']:
                        for tstamp in model_params['tstamp']:
                            for nrows in model_params['nrows']:
                                for test_lenght in model_params['test_lenght']:
                                    for nbins in model_params['nbins']:
                                        for n_estimators in model_params['n_estimators']:
                                            for bootstrap in model_params['bootstrap']:
                                                for class_weight in model_params['class_weight']:
                                                    for n_jobs in model_params['n_jobs']:
                                                        cmd  = 'python3 -u model_clf_rf.py --db_host=192.168.1.84'
                                                        cmd += ' --symbol='        + symbol
                                                        cmd += ' --interval='      + str(interval)
                                                        cmd += ' --shift='         + str(shift)
                                                        cmd += ' --X_symbols='     + X_symbols
                                                        cmd += ' --X_include='     + X_include
                                                        cmd += ' --X_exclude='     + X_exclude
                                                        cmd += ' --tstamp='        + str(tstamp).replace(' ', 'T')
                                                        cmd += ' --nrows='         + str(nrows)
                                                        cmd += ' --test_lenght='   + str(test_lenght)
                                                        cmd += ' --nbins='         + str(nbins)
                                                        cmd += ' --n_estimators='  + str(n_estimators)
                                                        cmd += ' --bootstrap='     + str(bootstrap)
                                                        cmd += ' --class_weight='  + str(class_weight)
                                                        cmd += ' --n_jobs='        + str(n_jobs)
                                                        cmd += '\n'
                                                        launcher_cmds.append(cmd)

shuffle(launcher_cmds)
fd = open('/home/selknam/var/scripts/model_clf_rf_gridsearch.1.bash', 'wt')
fd.writelines(launcher_cmds)
fd.close()
# %%

# conda activate rata310
# cd /home/selknam/dev/rata/src/
# nohup bash /home/selknam/var/scripts/model_clf_rf_gridsearch.1.bash &> /home/selknam/var/log/model_clf_rf_gridsearch.1.bash.log &
# tail -f /home/selknam/var/log/model_clf_rf_gridsearch.1.bash.log
# %%
