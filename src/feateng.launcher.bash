interval=$1
sleep 7
sleep $interval
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata310 && \
  python -u /home/selknam/dev/rata/src/feateng.launcher.py --conf=conf/rata.json --interval=$interval --nrows=350