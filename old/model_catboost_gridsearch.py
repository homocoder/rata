#%%
from random import shuffle
import pandas as pd

model_params = {
    'symbol'        : ['EURUSD'],
    'interval'      : [1, 3],
    'shift'         : [1, 3, 6, 9], # 1, 3, 6, 9
    'X_symbols'     : ['EURUSD'],
    'X_include'     : ['rsi,*'],
    'X_exclude'     : ['volatility_kcli'],
    'tstamp'        : pd.date_range(start='2022-07-19', periods=12, freq='6h'),
    'nrows'         : [3000],
    'test_lenght'   : [1000],
    'iterations'    : [1000],
    'learning_rate' : ['nn'],
    'depth'         : ['nn'],
    'l2_leaf_reg'   : ['nn'],
    'loss_function' : ['RMSE', 'MAPE'] # MAPE RMSE #['RMSE', 'MultiRMSE', 'MAE', 'Quantile', 'MAPE']
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
                                    for iterations in model_params['iterations']:
                                        for learning_rate in model_params['learning_rate']:
                                            for depth in model_params['depth']:
                                                for l2_leaf_reg in model_params['l2_leaf_reg']:
                                                    for loss_function in model_params['loss_function']:
                                                        cmd  = 'python3 -u model_catboost.py --db_host=192.168.1.83'
                                                        cmd += ' --symbol='        + symbol
                                                        cmd += ' --interval='      + str(interval)
                                                        cmd += ' --shift='         + str(shift)
                                                        cmd += ' --X_symbols='     + X_symbols
                                                        cmd += ' --X_include='     + X_include
                                                        cmd += ' --X_exclude='     + X_exclude
                                                        cmd += ' --tstamp='        + str(tstamp)
                                                        cmd += ' --nrows='         + str(nrows)
                                                        cmd += ' --test_lenght='   + str(test_lenght)
                                                        cmd += ' --iterations='    + str(iterations)
                                                        cmd += ' --learning_rate=' + str(learning_rate)
                                                        cmd += ' --depth='         + str(depth)
                                                        cmd += ' --l2_leaf_reg='   + str(l2_leaf_reg)
                                                        cmd += ' --loss_function=' + loss_function
                                                        cmd += '\n'
                                                        launcher_cmds.append(cmd)

shuffle(launcher_cmds)
fd = open('model_catboost_gridsearch.bash', 'wt')
fd.writelines(launcher_cmds)
fd.close()
# %%
# %%