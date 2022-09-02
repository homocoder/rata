# SQL delete from rates where True;
# SQL drop table feateng;

python -u rates_get_tstamp.py --db_host=192.168.1.83 --symbol=GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD,USDCHF,EURUSD --kind=forex --interval=1 --tstamp_end=2022-09-04 --periods=21
python -u rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=* --days=30
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=AUDUSD --kind=forex --interval=1 --nrows=12500  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=GBPUSD --kind=forex --interval=1 --nrows=12500 & 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=NZDUSD --kind=forex --interval=1 --nrows=12500  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=EURUSD --kind=forex --interval=1 --nrows=12500 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDCAD --kind=forex --interval=1 --nrows=12500  
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDJPY --kind=forex --interval=1 --nrows=12500 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDCHF --kind=forex --interval=1 --nrows=12500  

python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=AUDUSD --kind=forex --interval=3 --nrows=12500 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=GBPUSD --kind=forex --interval=3 --nrows=12500 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=NZDUSD --kind=forex --interval=3 --nrows=12500 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=EURUSD --kind=forex --interval=3 --nrows=12500 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDCAD --kind=forex --interval=3 --nrows=12500 &
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDJPY --kind=forex --interval=3 --nrows=12500 
python -u /home/selknam/dev/rata/src/feateng.py --db_host=192.168.1.83 --symbol=USDCHF --kind=forex --interval=3 --nrows=12500  

# nohup bash initialize.bash >& /home/selknam/var/log/initialize.bash.out &
