interval=$1
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python /home/selknam/dev/rata/src/rates_launcher.py \
         --db_conf=conf/db.json \
         --conf=conf/rates_launcher.$interval.json