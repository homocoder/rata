interval=$1
sleep 10
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py310 && \
  python -u /home/selknam/dev/rata/src/forecasts_binclf_launcher.py \
    --db_conf=conf/db.json \
    --symbol_conf=conf/rates_launcher.$interval.json \
    --forecasts_binclf_conf=conf/forecasts_binclf.json \
    #--forecast_datetime=2021-12-27T11:20:05 