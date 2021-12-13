rm /tmp/launch*
rm /tmp/xa*
cd /home/selknam/dev/rata/src
conda activate rata.py39
python /home/selknam/dev/rata/src/launch_prepare_and_model_binary_classifiers.py --db_conf=conf/db.json --symbol_conf=conf/update_ohlcv_and_forecast.5.forex.json --binary_classifiers_conf=conf/prepare_and_model_binary_classifiers.json
f=`ls /tmp/launch*`
cd /tmp
shuf $f > $f.shuf
split -n 2 $f.shuf
nohup bash /tmp/xaa > /home/selknam/var/log/xaa_clf.5.log &
nohup bash /tmp/xab > /home/selknam/var/log/xaa_clf.5.log &

