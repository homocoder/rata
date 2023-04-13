def generate_launchers():
#def generate_launchers(txt, txt_replace, interval, t1, t2):
    from datetime import datetime
    from sklearn.utils import shuffle
    import pandas as pd

    txt = """interval=5
    cd /home/selknam/dev/rata/src && \
    source /home/selknam/opt/miniconda3/bin/activate rata310 && \
    python -u /home/selknam/dev/rata/src/models_binclf_launcher.py \
        --db_conf=conf/db.json \
        --symbol_conf=conf/rates_launcher.$interval.json \
        --models_binclf_conf=conf/models_binclf.json \
        --model_datetime=0000-00-00T00:00:00
    """

    txt = """interval=5
    cd /home/selknam/dev/rata/src && \
    source /home/selknam/opt/miniconda3/bin/activate rata310 && \
    python -u /home/selknam/dev/rata/src/forecasts_binclf_launcher.py \
        --db_conf=conf/db.json \
        --symbol_conf=conf/rates_launcher.$interval.json \
        --forecasts_binclf_conf=conf/forecasts_binclf.json \
        --forecast_datetime=0000-00-00T00:00:00 
    """

    txt_replace = "0000-00-00T00:00:00"

    t1 = datetime(2021, 12, 29, 22, 00, 0)
    t2 = datetime(2022,  1,  6, 13, 40, 0) #jue 06 ene 2022 11:40:00 -03

    df = pd.DataFrame(pd.date_range(start=t1, end=t2, freq='5min'))
    #df = shuffle(df)
    cmd = ''
    for i in range(0, len(df)):
        cmd += txt.replace(txt_replace, str(df.iloc[i][0]).replace(' ', 'T'))

    return cmd