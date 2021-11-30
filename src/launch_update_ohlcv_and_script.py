# %%
from json import load

fd = open('conf/db.json', 'rt')
dbconf = load(fd)
fd.close()

db_params = ''
for i in dbconf:
    db_params += ' --' + i + '=' + dbconf[i].__str__()

fd = open('conf/update_ohlcv_and_forecast.forex.json', 'rt')
symbol_conf = load(fd)
fd.close()

cmd = ''
for i in symbol_conf:
    symbol_params = db_params
    for j in i:
        symbol_params += ' --' + j + '=' + i[j].__str__()
    cmd +='python /home/selknam/dev/rata/src/update_ohlcv_and_forecast.py ' + symbol_params + ' --kind=forex & \n'
print(cmd)

launch_file = '/tmp/launch_update_ohlcv_and_script.bash'

fd = open(launch_file, 'wt')
fd.write(cmd)
fd.close()

from subprocess import getoutput
#getoutput('bash -c "source /home/selknam/.bashrc &&  bash ' + launch_file + '"')

# %%
