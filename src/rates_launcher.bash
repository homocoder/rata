interval=$1
cd /home/selknam/dev/rata/src 
source /home/selknam/opt/miniconda3/bin/activate rata.py39 
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=USDJPY --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=GBPUSD --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=USDCAD --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=AUDUSD --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=EURUSD --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=USDCHF --days=1 &
  python3 rates_clean.py --db_host=192.168.1.83 --db_port=5432 --symbol=NZDUSD --days=1
  python -u /home/selknam/dev/rata/src/rates_launcher.py \
         --conf=conf/rata.json \
         --interval=$interval