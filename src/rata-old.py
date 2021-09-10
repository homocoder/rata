def main():
    pass

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['cholidb']
collection = db['choli-collection']


import datetime
post = {"author": "Mike",
    "text": "My first blog post!",
        "putalawealevel": 75,
    "tags": ["mongodb", "python", "pymongo"],
    "date": datetime.datetime.utcnow()}

posts = db.posts
post_id = posts.insert_one(post).inserted_id
post_id
print(db.list_collection_names())

# 🐭 LIB

class rata: # 🐭 
    def __init__(self):
        from pandas import DataFrame
        #Data
        self.df_rates = DataFrame()
        self.dataset = DataFrame()
        self.Xy = DataFrame()
        self.X = DataFrame()
        self.y = DataFrame()
        self.X_train = DataFrame()
        self.y_train = DataFrame()
        self.X_test = DataFrame()
        self.y_test = DataFrame()
        # Behaviour
        self.js_to_price = False
        self.price_to_js = False
        self.sY = list()
        self.sX = list()
        # Attributes
        self.symbols     = list()
        self.symbols_len = 0
        self.symbols_data_len = list()
        self.selected_features_names = list()
        # Config
        self.forex = True
        self.interval = '5m'
        self.include_all_indicators = False
        self.include_my_indicators = True
        self.datadir = 'ratadb'
        self.rates_file = 'default'
        self.back_data  = 900
        self.utc = -3
       
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.shuffle = False
        self.train_size = 0.8
        self.lag = -1
        self.rcut = 0
        self.profit_thresh = 0.15
        self.nsamples = 1000
        self.del_columns = ['symbol']
        self.target_class = '1B'
        self.target_feature = 'cat_roc3'
        self.macd_parms = '12_26'
        self.macd_filter = 3

        # Model Tuning
        self.n_estimators = 200  # xgboost, rf
        self.class_weight = 'balanced' # rf
        self.max_features = 60 # xgboost, rf
        self.row_weighting = True # xgboost
        self.n_steps = 5 # lstm

    #def get_rates(self):
    #    symbols = list(set([self.sY] + self.sX))
    #    df_rates = self.get_prices_multithreads(symbols[:])
    #    df_rates.set_index('tstamp', inplace=True)
    #    
    #    self.symbols_len = len(symbols)
    #    for s in symbols:
    #        self.symbols_data_len.append((s, len(df_rates[df_rates['symbol'] == s])))
    #    self.df_rates = df_rates        
    #    del df_rates
    #    self.symbols = symbols
 
    def get_csv_rates_myindicators(self):
        from pandas import read_csv
        import numpy as np
        if self.rates_file == 'default':
            rates_file  = self.datadir + '/rates/'
            rates_file += self.sY + '_' + self.interval + '.csv'
        else:
            rates_file  = self.datadir + '/rates/'
            rates_file += self.rates_file

        df = read_csv(rates_file, index_col='tstamp')
        df = df[['symbol', 'open', 'high', 'low', 'close', 'volume']].iloc[-self.back_data:]
        df = np.round(df, 5)
        df_rates = self.get_custom_indicators(df)
        symbols = [self.sY]

        self.symbols_len = len(symbols)
        for s in symbols:
            self.symbols_data_len.append((s, len(df_rates[df_rates['symbol'] == s])))
        self.df_rates = df_rates        
        del df_rates
        self.symbols = symbols
    
    def get_custom_indicators(self, df):
        import pandas as pd   
        import numpy as np   
        # TODO: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
        if self.include_all_indicators:
            import ta
            df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        if self.include_my_indicators:
            import ta
            MACD = ta.trend.MACD(close=df['close'], n_fast=12, n_slow=26, n_sign=9)
            macd        = pd.DataFrame(MACD.macd())
            macd_diff   = pd.DataFrame(MACD.macd_diff())
            macd_signal = pd.DataFrame(MACD.macd_signal())
            df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

            MACD = ta.trend.MACD(close=df['close'], n_fast=10, n_slow=21, n_sign=9) #10, 26, 8
            macd        = pd.DataFrame(MACD.macd())
            macd_diff   = pd.DataFrame(MACD.macd_diff())
            macd_signal = pd.DataFrame(MACD.macd_signal())
            df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)     

            MACD = ta.trend.MACD(close=df['close'], n_fast=7, n_slow=18, n_sign=7)
            macd        = pd.DataFrame(MACD.macd())
            macd_diff   = pd.DataFrame(MACD.macd_diff())
            macd_signal = pd.DataFrame(MACD.macd_signal())
            df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

            MACD = ta.trend.MACD(close=df['close'], n_fast=8, n_slow=21, n_sign=8)
            macd        = pd.DataFrame(MACD.macd())
            macd_diff   = pd.DataFrame(MACD.macd_diff())
            macd_signal = pd.DataFrame(MACD.macd_signal())
            df = pd.concat([df, macd_diff, macd_signal, macd], axis=1)

            KST      = ta.trend.KSTIndicator(close=df['close'])
            kst_sig  = pd.DataFrame(KST.kst_sig())
            kst      = pd.DataFrame(KST.kst())
            kst_diff = pd.DataFrame(KST.kst_diff())
            df = pd.concat([df, kst, kst_sig, kst_diff], axis=1)

            RSI = ta.momentum.rsi(close=df['close'])
            KAMA = ta.momentum.kama(close=df['close'])
            #OBV = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])
            df = pd.concat([df, RSI, KAMA], axis=1)

        Xcolumns = df.columns.tolist()
        Xcolumns.remove('symbol')
        for p in Xcolumns:
            for r in range(1, 12):
                df[p + 'roc' + str(r)] = df[p].pct_change(r).rolling(window=r).sum() * 100
            
            df[p + 'back1'] = df[p].shift(1)
            df[p + 'back2'] = df[p].shift(2)
            df[p + 'back3'] = df[p].shift(3)
            df[p + 'back4'] = df[p].shift(4)
            df[p + 'back5'] = df[p].shift(5)
            df[p + 'back6'] = df[p].shift(6)            
            
        if   self.target_feature == 'cat_roc1':
            df['cat_roc1']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc1', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc2':
            df['cat_roc2']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc2', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc3':
            df['cat_roc3']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc3', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc4':
            df['cat_roc4']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc4', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc5':
            df['cat_roc5']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc5', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc6':
            df['cat_roc6']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc6', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc7':
            df['cat_roc7']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc7', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc8':
            df['cat_roc8']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc8', self.profit_thresh), axis=1)
        elif self.target_feature == 'cat_roc9':
            df['cat_roc9']     = df.apply(lambda row : self.get_cat_roc(row, 'closeroc9', self.profit_thresh), axis=1)
        return(df)

    def get_cat_roc(self, row, column_name, threshold):
        roc = row[column_name]
        if   (self.target_class == '1B') and (roc >  threshold):            
            cat_roc = '1B'
        elif (self.target_class == '1S') and (roc < -threshold):
            cat_roc = '1S'
        else:
            cat_roc = '0Z'
        return(cat_roc)

    def get_dataset(self): # TODO: eliminar esto
        from pandas import to_datetime
        # Transponer los symbols como columnas
        df_merge = self.get_price_merge(df=self.df_rates, respuesta=self.sY, columns=self.df_rates.columns, dropna_thresh=100)

        # Eliminar algunas columnas
        eliminar = list()
        for s in self.symbols:
            for c in self.del_columns:
                eliminar.append('X_' + s + '_' + c)
        df_merge.drop(eliminar, axis=1, inplace=True)

        df_merge.index = to_datetime(df_merge.index)
        self.dataset = df_merge.copy()
        del df_merge
        
    def get_price_merge(self, df, respuesta, columns, dropna_thresh): # PARCHE HORRIBLE PERO FUNCA (4 Coronas en el cuerpo)
        df.columns = ['X_' + self.sY + '_' + c for c in df.columns]
        return(df)

    def get_price_merge_MALO(self, df, respuesta, columns, dropna_thresh):
        print(df)
        from functools import reduce
        import pandas as pd
        symbols   = df['symbol'].drop_duplicates().values.tolist()
        symbols.insert(0, symbols.pop(symbols.index(respuesta)))

        dfs = list()
        for s in symbols.copy():
            print(s)
            for c in columns:
                df_symbol = df[(df['symbol'] == s)][c]
                df_symbol.rename('X_' + s + '_' + c, inplace=True)
                if len(df_symbol) > dropna_thresh:
                    dfs.append(pd.DataFrame(df_symbol)) # ESTA PARTE DEL CODIGO SIEMPRE ESTUVO MALA!!!!!
                    print(len(dfs))
        print('antes')
        df_merge = reduce(lambda left, right: pd.merge(left, right, on='tstamp', how='left'), dfs)
        print('despues')
        print(df_merge)
        return(df_merge)

    def get_X_y(self):
        from sklearn.model_selection import train_test_split
        from pandas import isna
        
        target_feature = self.target_feature

        dataset = self.dataset.copy()
        X_forecast = dataset.drop([c for c in dataset.columns if '_roc' in c], axis=1).iloc[-1:, :] # Mismo criterio para X     
        columnX = 'X_' + self.sY + '_' + target_feature #columnX = 'X_SPY_cat_roc1'
        df = self.XtoY_lag(columnX, dataset, self.lag)
        df = df.iloc[90:, :] # Sacar los NaN que genera el rolling mean
        y_name = 'Y_' + self.sY + '_' + target_feature + '_L' + str(self.lag)
        y = df[y_name] # Y_BTCUSD_closeroc1_L-2
        to_drop = df.columns.to_list()
        to_drop.remove(y_name)
        Xy = df.drop([c for c in to_drop if '_roc' in c], axis=1) # Mismo criterio para X_forecast
        X = Xy.drop([y_name], axis=1) 

        # X.dropna(inplace=True, axis=0, how='any') ## TODO: imputar NaNs
        # Check fucking NANSSSSSSSSZZZZZZ
        #for j in X.columns:
        #    nnulls = 0            
        #    for i in range(0, len(df)):
        #        if isna(X[j].iloc[i]):
        #            nnulls += 1
        #    if nnulls:
        #        print(j, ' has ', nnulls, ' nulls')

        self.y = y
        self.X = X
        self.Xy = Xy
        self.X_forecast = X_forecast
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.train_size, shuffle=self.shuffle)
        del y, X, X_forecast, df

    def XtoY_lag(self, columnX, df, lag):
        newY = 'Y_' + columnX[2:] + '_L' + str(lag)
        df[newY] = df[columnX].shift(lag)
        df.dropna(inplace=True, how='all')
        df = df[ ~ (df[newY].isnull())]
        return(df)

    def get_signal_1B(self, df, row):
        Ycolumns = [i for i in df.columns if 'Y_' + self.sY + '_closeroc' in i]
        Ycolumns.sort()

        signal = '0'
        for i in Ycolumns:
            if row[i] < -self.stop_loss:
                signal = '0'
                break
            if row[i] > self.take_profit:
                signal = '1'
                break
        return(signal)

    def get_signal_1S(self, df, row):
        Ycolumns = [i for i in df.columns if 'Y_' + self.sY + '_closeroc' in i]
        Ycolumns.sort()

        signal = '0'
        for i in Ycolumns:
            if row[i] > self.stop_loss:
                signal = '0'
                break
            if row[i] < -self.take_profit:
                signal = '1'
                break
        return(signal)

    def signal_filter(self, df):
        diff = 'X_' + self.sY + '_MACD_diff_' + self.macd_parms
        #sign = 'X_' + self.sY + '_MACD_sign_' + self.macd_parms
        macd = 'X_' + self.sY + '_MACD_'      + self.macd_parms

        if self.macd_filter == 1:
            # Filter crossover
            dfB = df[(df[diff]          >  0.0) &                   
                    (df[macd]           <  0.0) &
                    (df[diff + 'back1'] <  0.0) &
                    (df[diff + 'back2'] < df[diff + 'back1']) &
                    (df[diff + 'back3'] < df[diff + 'back2']) &
                    (df[diff + 'back4'] < df[diff + 'back3']) &
                    (df[diff + 'back5'] < df[diff + 'back4'])]
            # Filter crossunder
            dfS = df[(df[diff]          <  0.0) & 
                    (df[macd]           >  0.0) &
                    (df[diff + 'back1'] >  0.0) &
                    (df[diff + 'back2'] > df[diff + 'back1']) &
                    (df[diff + 'back3'] > df[diff + 'back2']) &
                    (df[diff + 'back4'] > df[diff + 'back3']) &
                    (df[diff + 'back5'] > df[diff + 'back4'])]

        if self.macd_filter == 2:
            # Filter crossover
            dfB = df[(df[diff]          >  0.0) &
                    (df[macd]           <  0.0) &
                    (df[diff + 'back1'] <  0.0) &
                    (df[diff + 'back2'] < df[diff + 'back1']) &
                    (df[diff + 'back3'] < df[diff + 'back2']) &
                    (df[diff + 'back4'] < df[diff + 'back3'])]
            # Filter crossunder
            dfS = df[(df[diff]          <  0.0) &
                    (df[macd]           >  0.0) & 
                    (df[diff + 'back1'] >  0.0) &
                    (df[diff + 'back2'] > df[diff + 'back1']) &
                    (df[diff + 'back3'] > df[diff + 'back2']) &
                    (df[diff + 'back4'] > df[diff + 'back3'])]

        if self.macd_filter == 3:
            # Filter crossover
            dfB = df[(df[diff]          >  0.0) &                   
                    (df[macd]           <  0.0) &
                    (df[diff + 'back1'] <  0.0) &
                    (df[diff + 'back2'] < df[diff + 'back1']) &
                    (df[diff + 'back3'] < df[diff + 'back2']) ]
            # Filter crossunder
            dfS = df[(df[diff]          <  0.0) & 
                    (df[macd]           >  0.0) & 
                    (df[diff + 'back1'] >  0.0) &
                    (df[diff + 'back2'] > df[diff + 'back1']) &
                    (df[diff + 'back3'] > df[diff + 'back2']) ]

        if self.macd_filter == 4:
            # Filter crossover
            dfB = df[(df[diff]          >  0.0) &
                    (df[diff + 'back1'] <  0.0) &
                    (df[diff + 'back2'] < df[diff + 'back1']) &
                    (df[diff + 'back3'] < df[diff + 'back2']) &
                    (df[diff + 'back4'] < df[diff + 'back3'])]
            # Filter crossunder
            dfS = df[(df[diff]          <  0.0) &
                    (df[diff + 'back1'] >  0.0) &
                    (df[diff + 'back2'] > df[diff + 'back1']) &
                    (df[diff + 'back3'] > df[diff + 'back2']) &
                    (df[diff + 'back4'] > df[diff + 'back3'])]
        return(dfB, dfS)

    def signal_distribution(self, df, pfix=''):
        signal_distr = dict()
        signal_distr['total_samples'] = len(df)
        signal_distr['total_0Z_0Z']   = len(df[(df['signal_1B'] == '0') & (df['signal_1S'] == '0')])
        signal_distr['total_1B_0Z']   = len(df[(df['signal_1B'] == '1') & (df['signal_1S'] == '0')])
        signal_distr['total_0Z_1S']   = len(df[(df['signal_1B'] == '0') & (df['signal_1S'] == '1')])
        signal_distr['total_1B_1S']   = len(df[(df['signal_1B'] == '1') & (df['signal_1S'] == '1')])

        signal_distr['ratio_0Z_0Z']   = round((signal_distr['total_0Z_0Z'] / signal_distr['total_samples'] * 100), 1)
        signal_distr['ratio_1B_0Z']   = round((signal_distr['total_1B_0Z'] / signal_distr['total_samples'] * 100), 1)
        signal_distr['ratio_0Z_1S']   = round((signal_distr['total_0Z_1S'] / signal_distr['total_samples'] * 100), 1)
        signal_distr['ratio_1B_1S']   = round((signal_distr['total_1B_1S'] / signal_distr['total_samples'] * 100), 1)
        k = [i for i in signal_distr.keys()]
        for i in k: 
            signal_distr[pfix + i] = signal_distr.pop(i) 
        # TODO: avg trades per day
        return(signal_distr)

def find_arg(param):
    from sys import argv
    return([b for b in argv if param in b][0].split('=')[1])

def get_finnhub(symbol, interval, forex=True, crypto=False, exchange='OANDA', hours=100, token='boq33onrh5rdr1qtpl3g'):
    from datetime import datetime as dt
    import pandas as pd
    import requests
    
    symbolurl = symbol
    if forex:
        symbolurl = symbol[:3] + '_' + symbol[3:]    
    if exchange=='COINBASE':
        symbolurl = symbol[:3] + '-' + symbol[3:]
    
    symbolurl = exchange + ':' + symbolurl
    intervalurl = interval.replace('m', '')
    t0 = str(int((dt.now() - pd.Timedelta(hours=hours)).timestamp()))
    t1 = str(int((dt.now()).timestamp()))
    #t0 = str(int((dt.now() - pd.Timedelta(hours=800)).timestamp()))
    #t1 = str(int((dt.now() - pd.Timedelta(hours=400)).timestamp()))
    
    if forex:
        url = 'https://finnhub.io/api/v1/forex/candle?symbol='
    if crypto:
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
    df = df[['tstamp', 'unix_tstamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'status']].set_index(['tstamp']).sort_index()

    min_rates = int(df.iloc[-1:].index[0].isoformat()[14:16])
    min_now =   int(dt.now().isoformat()[14:16])
    if min_rates == min_now:
        if interval != '1m':
            df = df.iloc[:-1]
    print('min_rates,', min_rates)
    print('min_now,',   min_now)

    return(df)
