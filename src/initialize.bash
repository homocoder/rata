
#python -u rates_get_tstamp.py --db_host=192.168.1.84 --symbol=GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD,USDCHF,EURUSD --kind=forex --interval=1 --tstamp_end=2022-09-28 --periods=7
python -u rates_clean.py --db_host=192.168.1.84 --db_port=5432 --symbol=* --days=600
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=AUDUSD --kind=forex --interval=1 --nrows=13000  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=GBPUSD --kind=forex --interval=1 --nrows=13000 & 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=NZDUSD --kind=forex --interval=1 --nrows=13000  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=EURUSD --kind=forex --interval=1 --nrows=13000 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDCAD --kind=forex --interval=1 --nrows=13000  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDJPY --kind=forex --interval=1 --nrows=13000 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDCHF --kind=forex --interval=1 --nrows=13000  

python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=AUDUSD --kind=forex --interval=3 --nrows=13000 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=GBPUSD --kind=forex --interval=3 --nrows=13000 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=NZDUSD --kind=forex --interval=3 --nrows=13000 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=EURUSD --kind=forex --interval=3 --nrows=13000 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDCAD --kind=forex --interval=3 --nrows=13000 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDJPY --kind=forex --interval=3 --nrows=13000 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.84 --symbol=USDCHF --kind=forex --interval=3 --nrows=13000  

# delete from rates where True;
# drop table feateng;
# cd /home/selknam/dev/rata/src/
# nohup time bash initialize.bash >& /home/selknam/var/log/initialize.bash.out &
# tail -f /home/selknam/var/log/initialize.bash.out

