
def custom_resample_open(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike['open'].iloc[0]

def custom_resample_close(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike['close'].iloc[-1]

def custom_resample_volume(arraylike):
    from numpy import NaN
    if len(arraylike) == 0:
        return NaN
    else:
        return arraylike['volume'].drop_duplicates().sum()

def check_time_gaps(df, _conf):
    import pandas as pd
    print('\n##### Checking time gaps for: ', _conf['symbol'] + "' and r.interval=" + str(_conf['interval']), ' #####')
    df_diff_intervals = pd.DataFrame(df['tstamp'])
    df_diff_intervals['delta'] = df_diff_intervals['tstamp'] - df_diff_intervals['tstamp'].shift(-1)
    df_diff_intervals.set_index(df_diff_intervals['tstamp'], inplace=True, drop=True)
    df_diff_intervals['delta_minutes'] = df_diff_intervals['delta'].dt.total_seconds() / -60

    df_delta_minutes = df_diff_intervals['delta_minutes'][df_diff_intervals['delta_minutes'] > float(_conf['interval']) * 2]
    print('Len: ', len(df_delta_minutes))
    if len(df_delta_minutes > 0):
        print('First: ', df.iloc[0]['tstamp'])
        print('Last: ',  df.iloc[-1]['tstamp'])
    print('Gaps are in minutes. Showing gaps interval*3')
    for c in range(0, len(df_delta_minutes)):
        print(pd.DataFrame(df_delta_minutes).reset_index().iloc[c, 0], pd.DataFrame(df_delta_minutes).reset_index().iloc[c, 1])
    return(df_delta_minutes)

def read_tv_strategy(file_name):
    from unicodedata import normalize
    from pandas import read_csv
    import pandas as pd    
    
    df = read_csv(file_name)

    columns = list()
    for text in df.columns:
        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3
            pass
        for c in ['#', '/', '$', '%', ' ', '.', '-']:
            text = text.replace(c, '')
        text = normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8").lower()
        columns.append(text)

    strategy_name = file_name.split('_')[-4]

    df.columns = columns
    df = df[['datetime', 'trade', 'signal', 'type', 'price', 'profit']].reset_index(drop=True)
    df1 = df[df['type'].str.contains('Entry')].set_index('trade')
    df2 = df[df['type'].str.contains('Exit')].set_index('trade')
    df = df1.join(df2, rsuffix='_exit').reset_index()
    datetime_tmp = pd.to_datetime(df['datetime']).dt.round(freq='3min')

    for c in df.columns:
        df.rename({c: c + '_' + strategy_name}, axis=1, inplace=True)
    df['tstamp'] = datetime_tmp

    df.set_index(df['tstamp'], inplace=True)
    return df