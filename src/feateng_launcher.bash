sleep 11
cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/dbmaintain.py
  python -u /home/selknam/dev/rata/src/feateng_launcher.py --conf=conf/rata.json --interval=3
