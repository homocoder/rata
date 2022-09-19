# %%
from sys import argv
from rata.utils import parse_argv

fake_argv = 'rates_get_tstamp.py --db_host=192.168.1.84 --symbol=GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD,USDCHF,EURUSD --kind=forex --interval=1 --tstamp_end=2022-08-21 --periods=4'
fake_argv = fake_argv.split()
#argv = fake_argv #### *!
_conf = parse_argv(argv=argv)
_conf['symbol'] = _conf['symbol'].split(',')
_conf

#%%
import pandas as pd
cmd = ''
for s in _conf['symbol']:
    for d in pd.date_range(end=_conf['tstamp_end'], periods=_conf['periods']):
        cmd += 'python3 -u rates.py --db_host=' + _conf['db_host'] + ' --symbol=' + s + ' --kind=forex --interval=1 --tstamp=' + str(d)[:10]
        cmd += '; sleep 0.3 \n'
# %%
launch_file = '/home/selknam/var/scripts/rates_get_tstamp.bash'
with open(launch_file, 'wt') as fd:
    fd.write(cmd)

# %%
from subprocess import getoutput
print(getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"'))