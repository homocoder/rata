cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15 &

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD,AUDCHF \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15,AUDCHF_3_close_SROC_15 &

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD,NZDUSD \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15,NZDUSD_3_close_SROC_15 &

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD,GBPAUD \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15,GBPAUD_3_close_SROC_15 

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD,GBPNZD \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15,GBPNZD_3_close_SROC_15 &

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.3.113 --symbol=AUDUSD --interval=3 --nrows=4000 \
    --n_steps_in=60 --n_steps_out=3 --epochs=150 --test_lenght=1000 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD,EURGBP \
    --X_columns=tstamp,AUDUSD_3_close_SROC_15,EURGBP_3_close_SROC_15 &

    #--symbols=AUDUSD,AUDCHF,NZDUSD \
    #--X_columns=tstamp,AUDCHF_3_close_SROC_15,NZDUSD_3_close_SROC_15,AUDUSD_3_close_SROC_15,AUDUSD_3_trend_macd_diff 
    #--X_columns=tstamp,AUDUSD_3_trend_macd,AUDUSD_3_trend_macd_signal,AUDUSD_3_trend_macd_diff,AUDUSD_3_trend_macd_SROC_15,AUDUSD_3_trend_macd_signal_SROC_15,AUDUSD_3_trend_macd_diff_SROC_15,AUDUSD_3_close_SROC_15 
#symbols = ['AUDUSD', 'GBPAUD', 'AUDCHF', 'GBPNZD', 'AUDNZD', 'EURGBP', 'NZDUSD']
