cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.1.83 --symbol=AUDUSD --interval=3 --nrows=12000 \
    --n_steps_in=32 --n_steps_out=1 --epochs=60 --test_lenght=1500 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD \
    --X_columns=tstamp,AUDUSD_3_volatility_atr,AUDUSD_3_close_SROC_15,AUDUSD_3_close_SROC_12 &

cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.1.83 --symbol=AUDUSD --interval=3 --nrows=12000 \
    --n_steps_in=32 --n_steps_out=1 --epochs=120 --test_lenght=1500 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD \
    --X_columns=tstamp,AUDUSD_3_volatility_atr,AUDUSD_3_close_SROC_15,AUDUSD_3_close_SROC_12 &

    cd /home/selknam/dev/rata/src && \
  source /home/selknam/opt/miniconda3/bin/activate rata.py39 && \
  python -u /home/selknam/dev/rata/src/forecast.py --db_host=192.168.1.83 --symbol=AUDUSD --interval=3 --nrows=12000 \
    --n_steps_in=32 --n_steps_out=1 --epochs=300 --test_lenght=1500 \
    --y_column=y_AUDUSD_3_close_SROC_15_shift-15 \
    --symbols=AUDUSD \
    --X_columns=tstamp,AUDUSD_3_volatility_atr,AUDUSD_3_close_SROC_15,AUDUSD_3_close_SROC_12
