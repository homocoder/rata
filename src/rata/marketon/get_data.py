def get_finnhub(symbol='EURUSD', interval=5, kind='forex', exchange='OANDA', hours=100, tstamp=None, token='ca5qhiqad3ib7i7rp180'):
    """
    get ohlcv data from finnhub. returns a pd.DataFrame
    """
    from datetime import datetime as dt
    import pandas as pd
    import requests
    
    symbolurl = symbol
    if kind == 'forex':
        symbolurl = symbol[:3] + '_' + symbol[3:]
    if exchange=='COINBASE':
        symbolurl = symbol[:3] + '-' + symbol[3:]
    
    symbolurl = exchange + ':' + symbolurl
    interval = str(interval)
    intervalurl = interval.replace('m', '')
    
    if tstamp == None:
        t0 = str(int((dt.now() - pd.Timedelta(hours=hours)).timestamp()))
        t1 = str(int((dt.now()).timestamp()))
    else:
        tnow = tstamp
        t0 = str(int((tnow - pd.Timedelta(hours=hours)).timestamp()))
        t1 = str(int((tnow).timestamp()))
    
    if kind == 'forex':
        url = 'https://finnhub.io/api/v1/forex/candle?symbol='
    if kind == 'crypto':
        url = 'https://finnhub.io/api/v1/crypto/candle?symbol='
    url += symbolurl + '&resolution=' + intervalurl + '&from=' + t0 + '&to=' + t1 + '&token=' + token
    print(url)
    call = requests.get(url)

    if call.status_code != 200:
        print('Error when getting data from Finnhub')
        print('Error code is ' + str(call.status_code))
        print('Reason is ' + str(call.reason))
        call.close()
        call.raise_for_status()
    data = call.json()
    print('data_status,', data['s'])
    df = pd.DataFrame(data)
    df.rename(columns={"c": "close", "h": "high", "l": "low", "o": "open", "s": "status", "t": "unix_tstamp", "v": "volume"}, inplace=True)
    df['tstamp'] = pd.to_datetime(df['unix_tstamp'], unit='s')
    df['symbol'] = symbol
    df['interval'] = interval
    df['query_tstamp'] = dt.now()
    df = df[['query_tstamp', 'tstamp', 'unix_tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].set_index(['query_tstamp']).sort_index()

    min_rates = int(df.iloc[-1:].index[0].isoformat()[14:16])
    min_rates = int(df['tstamp'].iloc[-1].isoformat()[14:16])
    min_now =   int(dt.now().isoformat()[14:16])
    if min_rates == min_now:
        if interval != '1m':
            df = df.iloc[:-1]
    print('min_rates,', min_rates)
    print('min_now,',   min_now)

    return(df)