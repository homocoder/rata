interval=$1
shift=$2
my_precision=$3
sleep 30
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata310 && \
  python -u /home/selknam/dev/rata/src/predict_clf_rf.launcher.py --db_host=192.168.1.83 --symbol=EURUSD --interval=$interval --shift=$shift --my_precision=$my_precision