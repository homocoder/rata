interval=$1
cd /home/selknam/dev/rata/src 
source /home/selknam/opt/miniconda3/bin/activate rata310 
  python -u /home/selknam/dev/rata/src/rates.launcher.py \
         --conf=conf/rata.json \
         --interval=$interval