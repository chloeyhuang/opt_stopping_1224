import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import settings as settings

import math
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv
import datetime
from datetime import datetime, timedelta
import sympy

import scipy as sp
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, acf
import os
from tqdm import tqdm

header = settings.header
train_header = settings.train_header
train_suffix = settings.train_suffix
train_4h_suffix = settings.train_4h_suffix
tdiff = settings.tdiff

#   formatting for matplotlib graphs & pandas tables
plt.rcParams['text.usetex'] = settings.latex
plt.rcParams['figure.dpi'] = 200
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.titlepad'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["legend.loc"] = 'lower right'
pd.options.display.float_format = '{:,.5f}'.format

#   formatting into retrievable case data
files = np.array(os.listdir(train_header + train_suffix))
files_long = np.array(os.listdir(train_header + train_4h_suffix))

pos = files[np.nonzero(np.char.endswith(files, 'pos.csv'))]
td = files[np.nonzero(np.char.endswith(files, 'trade.csv'))]

#   gets the case number
getname = lambda x: int(x[7:-8]) if x[0] == "l" else int(x[5:-8])

getnametd = lambda x: int(x[7:-16]) if x[0] == "l" else int(x[5:-16])

pos = sorted(pos, key = getname)
td = sorted(td, key = getnametd)

pos_l = files_long[np.nonzero(np.char.endswith(files_long, 'pos.csv'))]
td_l = files_long[np.nonzero(np.char.endswith(files_long, 'trade.csv'))]

pos_l = sorted(pos_l, key = getname)
td_l = sorted(td_l, key = getnametd)

#   some util functions and variable definitions to help visualise data 
bid_prices = np.char.add("bid_p", (1 + np.arange(10)).astype(str))
bid_vols = np.char.add("bid_v", (1 + np.arange(10)).astype(str))

ask_prices = np.char.add("ask_p", (1 + np.arange(10)).astype(str))
ask_vols =  np.char.add("ask_v", (1 + np.arange(10)).astype(str))

#   util function for removing zeros/nans in pos dataframe
def pos_drop_zero(pos:pd.DataFrame):
    no_na = pos.dropna()
    return no_na[no_na['mid_p'] != 0]

#   converts csvs to pandas dataframes
def to_df(filename:str, l = False): 
    folder = train_header + train_4h_suffix if l == True else train_header + train_suffix
    df = pd.read_csv(folder + filename, parse_dates = ["timestamp"], index_col = 0, date_format = 'mixed')
    if filename[-7:-4] == 'pos':
        return pos_drop_zero(df)
    else:
        return df

def maketitle(x, td): 
    casenum = str(int(x[7:-16])) if x[0] == "l" else str(int(x[5:-16])) 
    return "buy case " + str(casenum) + "\n " + str(td.index[0])[:19] + " to " + str(str(td.index[-1])[:19])

#   graphs price figures with baseline buy times and sell times
def graph_price(pos_fname, td_fname, ret_fig = False):
    pos = to_df(pos_fname)
    td = to_df(td_fname)
    fig, ax = plt.subplots()
    ax.set(xmargin = 0, title = maketitle(td_fname, td), xlabel = "timestamp")
    ax.plot(time_match(pos, td).index, time_match(pos, td)["ask_p1"] * (0.9987), marker = "^", color = "red", markersize = '4', alpha = 0.2)
    ax.plot(sell_select(pos, td, tdiff).index, sell_select(pos, td, tdiff)["ask_p1"] * (1.0013), marker = "v", color = "green", markersize = '4', alpha = 0.2)
    ax.plot(td.index, td["ask_p1"])
    if ret_fig == True:
        return fig

#   gets the returns 
def get_returns(trade_data):
    tp = trade_data.index
    t = tp[0]
    ln = len(tp)

    bid_price = trade_data['bid_p1'][t:]
    ask_price = trade_data['ask_p1'][t]

    scaled_bid = (bid_price/ask_price)

    returns = np.zeros(len(scaled_bid))
    returns[0] = scaled_bid.iloc[0]-1
    returns[1:] = np.array(scaled_bid[1:])/np.array(scaled_bid[:-1]) - 1

    return pd.Series(data=returns, index = scaled_bid.index)

#   returns price - price.rolling(n).mean()
def get_rolling(trade_data, n = 500, log = False, interval = 1):
    data = trade_data[::interval]
    bid_price = data['bid_p1']
        
    if log == True:
        bid_price = np.log(bid_price)

    mean = bid_price.rolling(int(n/interval)).mean()
    diff = bid_price-mean

    return pd.DataFrame(data= {'price': bid_price, 'rolling_mean':mean , 'diff': diff}, index = bid_price.index)

#   calculates (vol weighted) score for given time / pos dataframe
def fast_mmtm(trade_data, pos_data, time = None, weight = 200):
    #   normalises weighted mean correctly
    norm_const = weight/10 *(1-np.exp(-1000/weight))

    def score(t):
        select = trade_data[:t]['ask_p1']
        times = select.index
        diff = trade_data['ask_p1'][t] - select
        tdiff = (times - t).to_numpy(dtype = 'float64')
            
        #   normalises time diff to be between 0 and 100 
        tdiff = -tdiff/tdiff[0]* 100
        gradient = (-diff/tdiff).dropna()
         #   returns momentum as sum of gradient * volume at time t * weighting
        #   where weighting exponentially decays; small weight = closer values weighted more
        return 10000 * np.sum(gradient * np.exp(-10 * np.abs(np.trim_zeros(tdiff)/weight)))/norm_const
    if time == None:
        return pd.Series(data = [score(t) for t in pos_data.index], index = pos_data.index)
    else:
        return score(time)

#   linear regression to get vol ratio
def vol_ratio_linreg(pre_vol, short_mom):
    x = 1/(1000 * pre_vol)
    y = -short_mom/10
    a = 0.128
    p = 1.174
    c = -0.106
    eq = (np.abs(a * x + (1-a) * y)**(p) + c * x)/100

    return 0.7604 + 20.534 * eq

#   estimates vol ratio after buying; pos data only used to get start / end time
def est_vol_ratio(trade_data, time):
    data = get_returns(trade_data)
    pre_vol = np.std(data[:time][-6000:])
    short_mom = fast_mmtm(trade_data, pos_data, start = True)

    return vol_ratio_linreg(pre_vol, short_mom)