import pandas as pd
from finta import TA


def TEMA_indicators(close_price):
    df = pd.DataFrame(list(close_price), columns=['close'])
    df['open'] = df.close
    df['high'] = df.close
    df['low'] = df.close
    df['volume'] = df.close
    # TEMA26 = TA.TEMA(df, 26)
    # TEMA26_slop = (TEMA26 - TEMA26.shift(1)) / (TEMA26.shift(1))
    # TEMA50 = TA.TEMA(df, 50)
    RSI = TA.RSI(df)
    MACD = TA.MACD(df)
    CCI = TA.CCI(df)
    ADX = TA.ADX(df)
    return  MACD.values.tolist()[-1:] + \
            RSI.values.tolist()[-1:] + \
            CCI.values.tolist()[-1:] + \
            ADX.values.tolist()[-1:] \
            # + TEMA26.values.tolist()[-1] + \
            # TEMA26_slop.values.tolist()[-1] + \
            # TEMA50.values.tolist()
