# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'update_ohlcv_and_forecast.py  --db_host=localhost --db_port=27017 --db_name=rata --symbol=EURUSD --interval=5 --kind=forex'
fake_argv = fake_argv.split()
#argv = fake_argv ####
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

id_xp = datetime.now().strftime('%Y%m%d-%H%M%S')
launch_file = '/tmp/launch_update_ohlcv_and_script.' + id_xp + '.' + _conf['conf'].split('/')[1] + '.bash'

fd = open(launch_file, 'wt')
fd.write(cmd)
fd.close()

#%%
from subprocess import getoutput
print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))

#%%
from os import remove
remove(launch_file)

# %%
