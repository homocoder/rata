# %%
from sys import argv
from rata.utils import parse_argv, load_conf

fake_argv = 'feateng_launcher.py --conf=conf/rata.json --interval=3 --nrows=500'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
conf  = load_conf(_conf['conf'])
_conf

# %%
db_params = '--db_host=' + conf['db_host']
symbol_conf = conf['symbols']
cmd = ''
for i in symbol_conf:
    symbol_params = db_params
    for j in i:
        symbol_params += ' --' + j + '=' + i[j].__str__()
    symbol_params += ' --interval=' + str(_conf['interval'])
    symbol_params += ' --nrows='    + str(_conf['nrows'])
    cmd += 'python -u /home/selknam/dev/rata/src/rates_clean.py --db_host=192.168.1.84 --symbol=' + i['symbol']+ ' --days=3 \n' #TODO:
    cmd += 'python -u /home/selknam/dev/rata/src/feateng.py ' + symbol_params + ' & \n'
print(cmd)

# %%
from datetime import datetime
id_xp = datetime.now().strftime('%Y%m%d-%H%M%S')
launch_file = '/home/selknam/var/scripts/feateng_launcher.' + id_xp + '.' + _conf['conf'].split('/')[1] + '.bash'

fd = open(launch_file, 'wt')
fd.write(cmd)
fd.close()

#%%
from subprocess import getoutput
print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))