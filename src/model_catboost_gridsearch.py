#%%
from random import shuffle
model_params = {
    'symbol'        : ['EURUSD'],
    'interval'      : [1, 3],
    'shift'         : [3, 6, 9], # 1, 3, 6, 9
    'X_symbols'     : ['EURUSD', 'EURUSD,GBPUSD', 'EURUSD,USDCAD'],
    'X_include'     : ['rsi,*'],
    'X_exclude'     : ['volatility_kcli'],
    'nrows'         : [3000],
    'test_lenght'   : [800],
    'iterations'    : [60, 90, 160],
    'learning_rate' : [0.03, 0.06, 0.09, 0.3],
    'depth'         : [1, 3, 12],
    'l2_leaf_reg'   : [1, 9, 30, 60],
    'loss_function' : ['RMSE', 'MultiRMSE', 'MAE', 'Quantile', 'MAPE']
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
                        for nrows in model_params['nrows']:
                            for test_lenght in model_params['test_lenght']:
                                for iterations in model_params['iterations']:
                                    for learning_rate in model_params['learning_rate']:
                                        for depth in model_params['depth']:
                                            for l2_leaf_reg in model_params['l2_leaf_reg']:
                                                for loss_function in model_params['loss_function']:
                                                    cmd  = 'python3 -u model_catboost.py --db_host=192.168.3.113'
                                                    cmd += ' --symbol='        + symbol
                                                    cmd += ' --interval='      + str(interval)
                                                    cmd += ' --shift='         + str(shift)
                                                    cmd += ' --X_symbols='     + X_symbols
                                                    cmd += ' --X_include='     + X_include
                                                    cmd += ' --X_exclude='     + X_exclude
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
