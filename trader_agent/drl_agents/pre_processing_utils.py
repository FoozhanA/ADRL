import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import datetime as dt
from finta import TA
import _pickle as cpickle
import glob
import os
from .utils import fill_missing_datetime_values
from ta.volatility import BollingerBands



class Read_Fill_Missing():
    def __init__(self,  raw_data, method = 'linear', cols2keep = ['datetime', 'stock', 'volume', 'accumulated_volume', 
                          'VWAP_per_candlestick', 'open', 'close', 'high', 'low', 'VWAP']): 
        self.cols2keep = cols2keep
        self.df = raw_data.loc[:, self.cols2keep]
        self.method = method

    def read_and_fill_missing_data(self):
        self.df = fill_missing_datetime_values(self.df, method = self.method, group_level = ['stock', 'date'])



class MakingTemaFeatures(): 
    def __init__(self, data, init_columns = ['datetime', 'stock', 'volume', 'accumulated_volume',
                                             'VWAP_per_candlestick', 'open', 'close', 'high', 'low', 'VWAP']): 
        self.init_columns = init_columns
#         self.df = data.loc[:, self.init_columns]
        self.df = data.copy()
    
        
    
    @staticmethod
    def TEMA_indicators(df):
        # s_TEMA, l_TEMA
        df['5TEMA'] = TA.TEMA(df, 5)
        df['26TEMA'] = TA.TEMA(df, 26)
        df['diff_5TEMA'] = df['5TEMA'] - df['5TEMA'].shift(1)
        df['der_5TEMA'] = np.sign(df['diff_5TEMA'])
        df['26TEMA_slop'] = (df['26TEMA'] - df['26TEMA'].shift(1)) / (df['26TEMA'].shift(1))
        df['diff_5_26_TEMA'] = df['5TEMA'] - df['26TEMA']
        df['sign_diff_5_26_TEMA'] = np.sign(df['diff_5_26_TEMA'])
        df['RSI'] = TA.RSI(df)
        df.loc[:, 'RSI_signal'] = np.where((df['RSI'] >= 70), 'sell', 'hold')
        df['RSI_signal'] = np.where((df['RSI'] <= 30), 'buy', df['RSI_signal'])
        df['HHV_RSI'] = df['RSI'].rolling(14).max()
        df['LLV_RSI'] = df['RSI'].rolling(14).min()
        df['HHV'] = df['high'].rolling(14).max()
        df['HHV_slope'] = (df['HHV'] - df['HHV'].shift(1)) / (df['HHV'].shift(1))
        df['LLV'] = df['low'].rolling(14).min()
        df['LLV_slope'] = (df['LLV'] - df['LLV'].shift(1)) / (df['LLV'].shift(1))
        df['RRI'] = (df.close - df.open) / (df.high - df.low + 0.0001)
        df_stock = pd.concat([TA.MACD(df)], axis=1)
        df = pd.concat([df, df_stock], axis=1)
        # TMACD features
        df['12TEMA'] = TA.TEMA(df, 12)
        df['TMACD'] = df['26TEMA'] - df['12TEMA']
        df['TMACD_signal_line'] = df['TMACD'].ewm(span=9).mean()
        df['diff_TMACD_line'] = df.TMACD - df['TMACD_signal_line']
        df['sign_TMACD_line'] = np.sign(df['diff_TMACD_line'])
        df['der_TMACD_diff'] = df['diff_TMACD_line'] - df['diff_TMACD_line'].shift(1)
        # MACD features
        df['diff_MACD_line'] = df['MACD'] - df['SIGNAL']
        df['der_MACD_diff'] = df['diff_MACD_line'] - df['diff_MACD_line'].shift(1)
        df['sign_MACD_line'] = np.sign(df['diff_MACD_line'])
        # TEMA features
        df.loc[:, 'direction'] = np.where((df['der_MACD_diff'] > 0), 'upward', 'downward')
        df.loc[:, 'signal'] = np.where(((df['sign_diff_5_26_TEMA'] == 1) &
                                        (df['direction'] == 'upward')), 'buy', 'hold')
        df.signal = np.where(((df['sign_diff_5_26_TEMA'] == -1) &
                              (df['direction'] == 'downward')), 'sell', df.signal)
        df['Risk_indicator'] = 1 - ((abs(df['RSI'] - 50) / 50) * (1 - ((df['MACD'] / (df['open'])) * 100)))
        # boiling bands features
        indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bbh'] = indicator_bb.bollinger_hband()
        df['bbl'] = indicator_bb.bollinger_lband()
        return df



    @staticmethod
    def EntryApproval(df):
        df['long_entry'] = None
        df['short_entry'] = None
        df['long_exit'] = None
        df['short_exit'] = None
        for i in reversed(range(1, len(df))):
            if (df.loc[df.index[i - 1], '26TEMA'] >= df.loc[df.index[i - 1], '5TEMA']
                    and df.loc[df.index[i], '5TEMA'] >= df.loc[df.index[i], '26TEMA']
                    and df.loc[df.index[i], 'direction'] == 'upward'
                    and i >= len(df) - 6
            ):
                df.loc[df.index[i], 'long_entry'] = 1
                df.loc[df.index[i], 'short_exit'] = 1
            if (df.loc[df.index[i - 1], '26TEMA'] <= df.loc[df.index[i - 1], '5TEMA']
                    and df.loc[df.index[i], '5TEMA'] <= df.loc[df.index[i], '26TEMA']
                    and df.loc[df.index[i], 'direction'] == 'downward'
                    and i >= len(df) - 6
            ):
                df.loc[df.index[i], 'short_entry'] = 1
                df.loc[df.index[i], 'long_exit'] = 1
            if (df.loc[df.index[i - 1], '26TEMA'] >= df.loc[df.index[i - 1], '5TEMA']
                    and df.loc[df.index[i], '5TEMA'] >= df.loc[df.index[i], '26TEMA']
                    and df.loc[df.index[i], 'direction'] == 'upward'
                    and i < len(df) - 6
            ):
                df.loc[df.index[i], 'long_entry'] = 1
                df.loc[df.index[i], 'short_exit'] = 1
                break
            if (df.loc[df.index[i - 1], '26TEMA'] <= df.loc[df.index[i - 1], '5TEMA']
                    and df.loc[df.index[i], '5TEMA'] <= df.loc[df.index[i], '26TEMA']
                    and df.loc[df.index[i], 'direction'] == 'downward'
                    and i < len(df) - 6
            ):
                df.loc[df.index[i], 'short_entry'] = 1
                df.loc[df.index[i], 'long_exit'] = 1
                break
        return df

    @staticmethod
    def zone(df):
        df['buy_zone'] = None
        df['sell_zone'] = None
        for idx in df.loc[df['long_entry'] == 1].index:
            df.loc[idx, 'buy_zone'] = 1
            for idx_1 in range(idx, len(df) - 1):
                if df.loc[idx_1, 'direction'] == 'upward' and df.loc[idx_1 + 1, 'direction'] == 'downward':
                    df.loc[idx_1 + 1, 'buy_zone'] = 1
                    break
                if (df.loc[idx_1, 'MACD'] < df.loc[idx_1, 'SIGNAL']
                        and df.loc[idx_1 + 1, 'MACD'] > df.loc[idx_1 + 1, 'SIGNAL']):
                    df.loc[idx_1 + 1, 'buy_zone'] = 1
                    break
                else:
                    df.loc[idx_1 + 1, 'buy_zone'] = 1
        for idx in df.loc[df['short_entry'] == 1].index:
            df.loc[idx, 'sell_zone'] = 1
            for idx_1 in range(idx, len(df) - 1):
                if df.loc[idx_1, 'direction'] == 'downward' and df.loc[idx_1 + 1, 'direction'] == 'upward':
                    df.loc[idx_1 + 1, 'sell_zone'] = 1
                    break
                if (df.loc[idx_1, 'MACD'] > df.loc[idx_1, 'SIGNAL']
                        and df.loc[idx_1 + 1, 'MACD'] < df.loc[idx_1 + 1, 'SIGNAL']):
                    df.loc[idx_1 + 1, 'sell_zone'] = 1
                    break
                else:
                    df.loc[idx_1 + 1, 'sell_zone'] = 1
        df.loc[:, 'zone'] = np.where((df.buy_zone == 1), 'buy', 'hold')
        df['zone'] = np.where((df.sell_zone == 1), 'sell', df.zone)
        return df
    
    
    @staticmethod
    def remove_extra_cols(df):
        df.drop(['short_entry', 'long_entry','buy_zone', 'sell_zone'], axis =1, inplace = True)
        df.fillna(0, inplace = True)
        return df
    
    def from_raw_to_tema_features(self):
        self.df = self.TEMA_indicators(self.df)
        self.df = self.EntryApproval(self.df)
        self.df = self.zone(self.df)
        self.df = self.remove_extra_cols(self.df)
        return self.df
    


class Data_Preprocessing():
    def __init__(self, df,  remove_duplicates , stock, act_price  , init_cols, columns_to_env, extra_columns, 
                 make_validation, test_ratio , valid_ratio,  train_mins_in_episode, 
                 valid_mins_in_episode, test_mins_in_episode, path2scaler, 
                 path2scaledColumns,  rd_shuffle_seed ):
        """
        mode can take two values: inference, training
        """
        
        self.stock = stock
        self.df = df.loc[df.stock == self.stock, :]
        self.remove_duplicates = remove_duplicates
        self.act_price = act_price 
        self.init_cols = init_cols
        self.columns_to_env = columns_to_env
        self.columns_to_scale = []
        self.extra_columns  = extra_columns
        self.make_validation =make_validation 
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.rd_shuffle_seed = rd_shuffle_seed
        self.train_mins_in_episode = train_mins_in_episode
        self.valid_mins_in_episode = valid_mins_in_episode
        self.test_mins_in_episode  = test_mins_in_episode
        self.path2scaler = path2scaler
        self.path2scaledColumns = path2scaledColumns

    
        
    def treat_duplicates(self):
        smdf = self.df.loc[:, ['stock', 'datetime']]
        len1 = len(smdf)
        smdf.drop_duplicates(inplace = True)
        len2= len(smdf)
        print(f'There is {len1 - len2} duplicates in date-time \n')
        if self.remove_duplicates: 
            self.df = self.df.groupby(['stock', 'datetime'], as_index = False).apply(lambda z: z.iloc[0, :])
            print('Duplicates are removed! \n')
        return True
        
    def get_missing_datetimes(self):
    
        def make_dt_frame(mn_time, mx_time, unique_dates):
            for date_nb , date in enumerate(unique_dates):
                strt = dt.datetime.combine(date, mn_time)
                ed = dt.datetime.combine(date, mx_time)
                if date_nb ==0: 
                    all_dt = list(pd.date_range(start = strt, end = ed, freq = 'min'))
                else: 
                    all_dt = all_dt+ list(pd.date_range(start = strt, end = ed, freq = 'min')) 
            dt_frame = pd.DataFrame()
            dt_frame['datetime'] = pd.to_datetime(all_dt)
            return dt_frame


        smdf= self.df.loc[:, ['stock', 'datetime']]
        smdf.sort_values(by = ['stock', 'datetime'], inplace = True)

        smdf.datetime = pd.to_datetime(smdf.datetime)
        smdf['hour'] = smdf.datetime.dt.hour
        smdf['date'] = smdf.datetime.dt.date
        mn_time = pd.to_datetime('2000-01-01 09:30:00').time()
        mx_time = pd.to_datetime('2000-01-01 16:00:00').time()
        smdf['dummy'] =1

        unique_dates = list(smdf.date.unique())
        dt_frame = make_dt_frame(mn_time, mx_time, unique_dates)
        dt_frame['stock'] = self.stock
        dt_frame = dt_frame.loc[:, ['stock', 'datetime']]
        dt_frame.sort_values(by = ['stock', 'datetime'], inplace = True)
            
        dt_frame.index = range(len(dt_frame))
        dt_frame['dummy']  = 1
        dt_frame['date'] = dt_frame.datetime.dt.date 
        dt_frame['hour'] = dt_frame.datetime.dt.hour
        all_datetime = dt_frame.merge(smdf, on = ['stock', 'date', 'datetime', 'hour'], how = 'left')

        nan_df = all_datetime.isnull()
        nan_df = nan_df.any(axis=1)
        nan_df = all_datetime[nan_df]

        self.missing_df = nan_df.groupby(['stock', 'date', 'hour'], as_index = False).agg({'dummy_x':'sum'})
        self.missing_df.rename({'dummy_x': 'nb_missing'}, axis =1, inplace = True)
        print(f'Missing times per day-hour, max = {self.missing_df.nb_missing.max()}: \n')
#         print(self.missing_df)
        print()
        return True
    
    
    def split_train_valid_test(self):
        
        dates = list(self.df.datetime.dt.date.unique())
        dates.sort()
        tn = int(len(dates)*(1-self.test_ratio))
        vn = int(len(dates)*(1-self.test_ratio - self.valid_ratio))
        if self.rd_shuffle_seed is not None:
            if self.rd_shuffle_seed !=0: 
                random.Random(self.rd_shuffle_seed).shuffle(dates)
        else: 
            random.shuffle(dates)

        overlap = 0
        if (len(dates)-tn > 2 ):
            overlap = (len(dates)-tn)*2//3
        self.train_dates, self.valid_dates, self.test_dates = dates[0:vn], dates[vn:tn], dates[-1:]
        # self.train_dates, self.valid_dates, self.test_dates = dates[0:vn] , dates[vn:tn]   , dates[tn:]
        nb_dates = list(map(lambda z: len(z), [self.train_dates, self.valid_dates, self.test_dates]))
        print(f'Namber of days in train, valid and test = {nb_dates}. \n')

        self.train = self.df.loc[self.df.date.isin(self.train_dates), :]
        if self.make_validation:
            self.valid = self.df.loc[self.df.date.isin(self.valid_dates), :]
        self.test = self.df.loc[self.df.date.isin(self.test_dates), :]
        return True

    

        
    @staticmethod
    def add_episode(df, mins_in_episode): 
        def aux_episod(z):
            if mins_in_episode is not None and mins_in_episode != -1:
                z['idd'] = range(len(z)-1, -1, -1)
                z.end_episod= ((z.idd % mins_in_episode) ==0)
                z.drop('idd', axis =1, inplace = True)
            elif mins_in_episode is None:
                z.end_episod.iloc[-1] = True
            else:
                mask = pd.to_datetime(z.datetime).dt.day != pd.to_datetime(z.datetime.shift(-1)).dt.day
                z.end_episod.iloc[mask] = True

            return z

        df = df.sort_values(by = ['stock', 'datetime'], axis =0, ignore_index = True)
        df['end_episod'] = False
        df = df.groupby(['date'], as_index = False).apply(lambda z: aux_episod(z)) 
        return df
        
    
    def cat_to_num(self):   ## N. 1
        self.type_change_dict = {}
        self.type_change_dict['zone'] = {'buy': -0.5, 'hold':0, 'sell':0.5}
        self.type_change_dict['signal'] = {'buy': -0.5, 'hold':0,  'sell':0.5}
        self.type_change_dict['direction'] = {'downward':-0.5, 'upward':0.5}
        self.type_change_dict['RSI_signal'] = {'hold':0, 'buy':-0.5, 'sell': 0.5}
        for col in self.type_change_dict.keys(): 
            self.df[col] = self.df.loc[:, col].map(self.type_change_dict[col])


    
    def basic_preproccessing(self, dropna= True): ## N. 2

        self.df['stock_index'] = 1

        def add_next_price(z): 
            z['next_' + self.act_price] = z[self.act_price].shift(-1)   
            return z


        if 'timestamp' in self.df.columns: 
            self.df.rename({'timestamp':'datetime', 'action': 'action_1'}, axis =1, inplace = True)
        else: 
            self.df.rename({'action': 'action_1'}, axis =1, inplace = True)
        self.df['datetime'] = pd.to_datetime(self.df.datetime)
        self.df.sort_values(by = ['stock', 'datetime'], axis =0, inplace = True, ignore_index= True)
        
        
        self.df['nscld_'+self.act_price] = self.df.loc[:, self.act_price]
        
        self.df['next_' + self.act_price] = None  
        self.df['date'] = self.df.datetime.dt.date
        self.df = self.df.groupby(['stock', 'date'], as_index = False).apply(lambda z: add_next_price(z))
        print(self.df.isna().sum())
        print('###################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', self.df.shape)
        if dropna:
            self.df.dropna(inplace = True)
        else: 
            self.df.fillna(1, inplace = True)
        print('###################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', self.df.shape)

        self.df.index = range(len(self.df))
        self.df.sort_values(by = ['stock', 'date', 'datetime'], inplace = True, ignore_index= True, axis=0)
        self.df = self.df.loc[:, self.init_cols+ ['date'] + self.columns_to_env + self.extra_columns]

        assert len(self.df)>0, 'There is no data here!'
        return True 


    def _in_where(self,n):
        if n in self.test_dates:
            return 'test'
        elif n in self.train_dates:
            return 'train'
        else:
            return 'valid'

    def scale_train_data(self):
        for col in self.columns_to_env:
            if self.train[col].abs().max() > 1.3:
                self.columns_to_scale.append(col)

        # self.scaler = MinMaxScaler(feature_range=(0.0, 0.8))
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train[self.columns_to_scale])
        self.train.loc[:, self.columns_to_scale] = self.scaler.transform(self.train.loc[:, self.columns_to_scale])
        # print(self.train.columns)
        # print(self.train.close.max(),self.train.close.min())
        # print(self.train.stock.unique())
        self.test.loc[:, self.columns_to_scale] = self.scaler.transform(self.test.loc[:, self.columns_to_scale])
        if self.make_validation:
            self.valid.loc[:, self.columns_to_scale] = self.scaler.transform(self.valid.loc[:, self.columns_to_scale])
        # print(self.test.close.max(),self.test.close.min())
        # print(self.valid.close.max(),self.valid.close.min())


        with open(self.path2scaler, "wb") as output_file:
            cpickle.dump(self.scaler, output_file)

        with open(self.path2scaledColumns, "wb") as output_file:
            cpickle.dump(self.columns_to_scale, output_file)
        self.df['where'] = self.df.date.apply(lambda z: self._in_where(z))

        self.df.index = range(len(self.df))

    def scale_infer_data(self):
        with open(self.path2scaler, "rb") as input_file:
            self.scaler = cpickle.load(input_file)

        with open(self.path2scaledColumns, "rb") as input_file:
            self.columns_to_scale = cpickle.load(input_file)

        self.test.loc[:, self.columns_to_scale] = self.scaler.transform(self.test.loc[:, self.columns_to_scale])
        self.df['where'] = self.df.date.apply(lambda z: self._in_where(z))

        self.df.index = range(len(self.df))



    def add_episodetotrain(self):   ## N. 4

        self.train = self.add_episode(self.train, mins_in_episode = self.train_mins_in_episode)

        if self.make_validation:
            self.valid = self.add_episode(self.valid, mins_in_episode = self.valid_mins_in_episode)
        self.test = self.add_episode(self.test,  mins_in_episode = self.test_mins_in_episode)
        return True

    def add_episodetoinfer(self):
        self.test = self.add_episode(self.test, mins_in_episode=self.test_mins_in_episode)
        return True

    def data_to_taining_model(self):
        self.cat_to_num()
        self.basic_preproccessing()
        self.split_train_valid_test()
        self.scale_train_data()
        assert (self.train.isna().sum().sum()+self.test.isna().sum().sum() == 0), 'There is nas in train, valid or in test data'
        self.add_episodetotrain()
        sm = 0
        lm = True
        lst2check = [self.df, self.train, self.valid, self.test]
        for data in lst2check:
            sm += data.isna().sum().sum()
            lm = (lm and len(data)>0)
        assert sm ==0, 'There is a problem here!'
        assert lm ==True, 'There is a problem here, some data is empty!'
        print('Pre-processing is done and data in ready for the model!')

    def data_to_inference_model(self):
        self.test_ratio = 1
        self.valid_ratio = 0
        self.cat_to_num()
        self.basic_preproccessing(dropna=False)
        self.split_train_valid_test()
        self.scale_infer_data()
        assert (self.train.isna().sum().sum()+self.test.isna().sum().sum() == 0), 'There is nas in train, valid or in test data'
        self.add_episodetoinfer()
        sm = 0
        lm = True
        lst2check = [self.df, self.test]
        for data in lst2check:
            sm += data.isna().sum().sum()
            lm = (lm and len(data)>0)
        assert sm ==0, 'There is a problem here!'
        assert lm ==True, 'There is a problem here, some data is empty!'
        print('Pre-processing is done and data in ready for the model!')





def get_melted_preds(df, stk):
    tgts = ['close', 'high', 'low', 'mhl', 'vwap']
    data = df[df.stock == stk]
    for tt, tgt in enumerate(tgts): 
        smdf = data[data.target_var == tgt]
        RFM = Read_Fill_Missing(smdf, main_method= 'ffill', limit_direction= 'forward', 
                                cols2keep = ['datetime', 'stock', 'target_var', 'pred_prob_1'])
        RFM.read_and_fill_missing_data()
        smdf = RFM.df
        smdf.rename({'pred_prob_1': f'{tgt}_pred_prob_1'}, axis =1, inplace = True)
        smdf.drop('target_var', axis =1, inplace = True)
        if tt ==0: 
            SMDF = smdf
        else: 
            SMDF = SMDF.merge(smdf, on = ['datetime', 'stock'], how = 'outer')
    return SMDF 

def read_dir_preds(path2dirpreds_folder): 
    all_files = glob.glob(os.path.join(path2dirpreds_folder, "*"))
    all_files = [x for x in all_files if '_1' in x]
    for nn, ff in enumerate(all_files):
        aux_df = pd.read_csv(ff)
        if nn ==0: 
            df = aux_df
        else:
            df = pd.concat([df, aux_df], ignore_index = True)

    cols2keep = ['datetime', 'stock', 'target_var','pred_prob_1']
    df = df.loc[:, cols2keep]
    df.target_var = df.target_var.apply(lambda x: 'mhl' if x == 'mean-high-low' else x)
    return df