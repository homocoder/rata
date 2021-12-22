interval=$1
sleep 15
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python /home/selknam/dev/rata/src/forecasts_binclf_launcher.py \
    --db_conf=conf/db.json \
    --symbol_conf=conf/rates_launcher.$interval.json \
    --forecasts_binclf_conf=conf/forecasts_binclf.json