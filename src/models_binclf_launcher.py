# %%
from sys import argv
from rata.utils import parse_argv

fake_argv  = 'models_binclf.py --db_conf=conf/db.json '
fake_argv += ' --symbol_conf=conf/rates_launcher.5.json '
fake_argv += ' --models_binclf_conf=conf/models_binclf.json '
fake_argv += ' --models_datetime=2021-12-01T00:01:05 '
fake_argv = fake_argv.split()
#argv = fake_argv #### !
_conf = parse_argv(argv=argv)
_conf

# %%
from json import load
from datetime import datetime

fd = open(_conf['db_conf'], 'rt')
dbconf = load(fd)
fd.close()

db_params = ''
for i in dbconf:
    db_params += ' --' + i + '=' + dbconf[i].__str__()

fd = open(_conf['symbol_conf'], 'rt')
symbol_conf = load(fd)
fd.close()

cmd = ''
for i in symbol_conf:
    symbol_params = db_params
    for j in i:
        symbol_params += ' --' + j + '=' + i[j].__str__()
    cmd += 'time python -u /home/selknam/dev/rata/src/models_binclf.py ' + symbol_params + '  \n'
cmd = cmd.split('\n')[:-1]

fd = open(_conf['models_binclf_conf'], 'rt')
binary_classifiers_conf = load(fd)
fd.close()

if 'model_datetime' in _conf:
    binary_classifiers_conf[0]['model_datetime'] = [str(_conf['model_datetime']).replace(' ', 'T')]

for x in cmd:
    cmd2 = x

hyper = list()
for i in binary_classifiers_conf:
    #print(cmd2, i)
    for j in i:
        params = list()
        for k in i[j]:
            params.append(' --' + j + '=' + str(k))
        hyper.append(params)
  
import itertools
hyper = list(itertools.product(*hyper))

cmd2 = ''
for c in cmd:               
    for i in hyper:
        params = ''
        for j in i:
            params += j
        cmd2 += c +  params + ' \n'

id_xp = datetime.now().strftime('%Y%m%d-%H%M%S')
launch_file = '/home/selknam/var/scripts/models_binclf_launcher.' + id_xp + '.' + _conf['symbol_conf'].split('/')[1] + '.bash'

fd = open(launch_file, 'wt')
fd.write(cmd2)
fd.close()

import random
lines = open(launch_file, 'rt').readlines()
random.shuffle(lines)
open(launch_file, 'wt').writelines(lines)

#%%
from subprocess import getoutput
print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))