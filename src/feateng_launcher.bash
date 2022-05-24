sleep 16
cd /home/selknam/var && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py310 && \
  time python -u /home/selknam/dev/rata/src/feateng.py
date
