# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:32:18 2022

@author: iphon
"""

import pandas as pd
import yfinance as yf # You probably have to install it,
# either via "pip install yfinance" in cmd or if you're using anaconda
# with "conda install -c conda-forge yfinance" in anaconda cmd

def forex (ticker_strings, period ='1d', interval = '1d'):
    df_list = list()
    for ticker in ticker_strings:
        data = yf.download(ticker, group_by="ticker", period = period, interval = interval)
        data['ticker'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
        df_list.append(data)
     
    df = pd.concat(df_list)
    return(df)

tickers = ['USDCHF=X', 'EURUSD=X']
period = 'max'
# Yahoo Finance only has data, monthly and daily, till on average start of 2003 
df = forex(ticker_strings = tickers, period = period)
# https://pypi.org/project/yfinance/
# If you want to play around with the arguments in yf.download,
# the documentation is there.

