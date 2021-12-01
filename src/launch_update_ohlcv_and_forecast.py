# %%
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element) -> bool:
    try:
        int(element)
        return True
    except ValueError:
        return False

# %%
from sys import argv

fake_argv = 'launch_update_ohlcv_and_forecast.py --db_conf=conf/db.json --conf=conf/update_ohlcv_and_forecast.1.forex.json'

fake_argv = fake_argv.split()

#argv = fake_argv ####

_conf = dict()
for i in argv[1:]:
    if '=' in i:
        param = i.split('=')
        _conf[param[0].replace('--', '')] = param[1]

for i in _conf:
    b = _conf[i]
    if   b == 'True':
        _conf[i] = True
    elif b == 'False':
        _conf[i] = False
    elif is_int(b):
        _conf[i] = int(b)
    elif is_float(b):
        _conf[i] = float(b)
_conf

# %%
from json import load

fd = open(_conf['db_conf'], 'rt')
dbconf = load(fd)
fd.close()

db_params = ''
for i in dbconf:
    db_params += ' --' + i + '=' + dbconf[i].__str__()

fd = open(_conf['conf'], 'rt')
symbol_conf = load(fd)
fd.close()

cmd = ''
for i in symbol_conf:
    symbol_params = db_params
    for j in i:
        symbol_params += ' --' + j + '=' + i[j].__str__()
    cmd += 'python /home/selknam/dev/rata/src/update_ohlcv_and_forecast.py ' + symbol_params + ' --kind=forex & \n'
print(cmd)

from datetime import datetime
id_xp = datetime.now().strftime('%Y%m%d-%H%M%S')
launch_file = '/tmp/launch_update_ohlcv_and_script.' + id_xp + '.' + _conf['conf'].split('/')[1] + '.bash'

fd = open(launch_file, 'wt')
fd.write(cmd)
fd.close()
## TODO: DELETE TMP FILE

from subprocess import getoutput
print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))

from os import remove
remove(launch_file)

# %%
