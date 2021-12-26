# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates.py --db_conf=conf/db.json --conf=conf/rates_launcher.5.json '
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
    cmd += 'timeout 5 python -u /home/selknam/dev/rata/src/rates.py ' + symbol_params + ' & \n'
print(cmd)

id_xp = datetime.now().strftime('%Y%m%d-%H%M%S')
launch_file = '/home/selknam/var/scripts/rates_launcher.' + id_xp + '.' + _conf['conf'].split('/')[1] + '.bash'

fd = open(launch_file, 'wt')
fd.write(cmd)
fd.close()

#%%
from subprocess import getoutput
#print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))