interval=$1
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py310 && \
  python -u /home/selknam/dev/rata/src/rates_launcher.py \
         --conf=conf/rata.json \
         --interval=$interval
  #python -u /home/selknam/dev/rata/src/dbmaintain.py