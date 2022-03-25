import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import datetime as dt
from finta import TA
import _pickle as cpickle
from ta.volatility import BollingerBands

def TEMA_indicators(df):
    df['26TEMA'] = TA.TEMA(df, 26)
    df['26TEMA_slop'] = (df['26TEMA'] - df['26TEMA'].shift(1)) / (df['26TEMA'].shift(1))
    df['RSI'] = TA.RSI(df)
    df.loc[:, 'RSI_signal'] = np.where((df['RSI'] >= 70), 'sell', 'hold')
    df['RSI_signal'] = np.where((df['RSI'] <= 30), 'buy', df['RSI_signal'])
    df_stock = pd.concat([TA.MACD(df)], axis=1)
    df = pd.concat([df, df_stock], axis=1)
    # TMACD features
    df['50TEMA'] = TA.TEMA(df, 50)
    df['50TEMA_slop'] = (df['50TEMA'] - df['50TEMA'].shift(1)) / (df['50TEMA'].shift(1))
    df['TMACD'] = df['50TEMA'] - df['26TEMA']
    # TEMA features
    df.loc[:, 'direction'] = np.where((df['der_MACD_diff'] > 0), 'upward', 'downward')
    df['Risk_indicator'] = 1 - ((abs(df['RSI'] - 50) / 50) * (1 - ((df['MACD'] / (df['open'])) * 100)))

    indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bbh'] = indicator_bb.bollinger_hband()
    df['bbl'] = indicator_bb.bollinger_lband()
    return df
