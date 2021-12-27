interval=$1
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  timeout 3540 \
  python /home/selknam/dev/rata/src/models_binclf_launcher.py \
    --db_conf=conf/db.json \
    --symbol_conf=conf/rates_launcher.$interval.json \
    --models_binclf_conf=conf/models_binclf.json
    #--model_datetime=2021-12-01T00:01:05 