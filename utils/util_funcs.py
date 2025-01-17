import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import utils.settings as settings

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

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=10)

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

#   a bunch of util functions and variable definitions to help visualise data 
bid_prices = np.char.add("bid_p", (1 + np.arange(10)).astype(str))
bid_vols = np.char.add("bid_v", (1 + np.arange(10)).astype(str))

ask_prices = np.char.add("ask_p", (1 + np.arange(10)).astype(str))
ask_vols =  np.char.add("ask_v", (1 + np.arange(10)).astype(str))

#   converts all the csvs into dataframes
def pos_drop_zero(pos:pd.DataFrame):
    no_na = pos.dropna()
    return no_na[no_na['mid_p'] != 0]

def to_df(filename:str, l = False): 
    folder = train_header + train_4h_suffix if l == True else train_header + train_suffix
    df = pd.read_csv(folder + filename, parse_dates = ["timestamp"], index_col = 0, date_format = 'mixed')
    if filename[-7:-4] == 'pos':
        return pos_drop_zero(df)
    else:
        return df

def to_df_no_ts(filename:str, l = False): 
    folder = train_header + train_4h_suffix if l == True else train_header + train_suffix
    return pd.read_csv(folder + filename, parse_dates = ["timestamp"], date_format = 'mixed')

#   matches the times where trading occurs with the times specified in pos data; just for easy access (potentially not useful? )
time_match = lambda pos, td: td[td.index.isin(pos.index)]

#   selects the trade data of time tdiff after when the assets were bought according to the pos data 
def sell_select(pos, td, tdiff):
    new_times = pos.index + timedelta(seconds=tdiff)
    return td[td.index.isin(new_times)]

#   calculates pnl
#   fills buy/sell order at time t given volume v, filling out ask/bid price in increasing/decreasing order; return vol at each price, money spent at each price, and total money spent/earned
#   tl;dr assuming market orders of volume vol; and op ~ operation: 1 = sell, -1 = buy 
def fulfill_order(td:pd.DataFrame, t:pd.Timestamp, vol:int, op:int, add_info = False):
    t_n = td[t:].index[0]

    if t != t_n:
        print("There was no data at {0}. Order was executed at the next time with data at {1}".format(t, t_n))
    
    d = td.at_time(t_n)

    prices = []
    vols = []
    if op == -1: 
        prices = d[ask_prices].values.flatten()
        vols = abs(d[ask_vols].values.flatten())
    if op == 1:
        prices = d[bid_prices].values.flatten()
        vols = abs(d[bid_vols].values.flatten())
    
    vol_left = vol
    counter = 0
    bins = np.zeros(10)
    #   better way of doing this bit? or not going to affect speed much since max 10 bins
    while vol_left > 0 and counter < 10: 
        v = min(vol_left, vols[counter])
        bins[counter] = v
        vol_left = vol_left - v
        counter += 1 
    
    if vol_left > 0:
        raise Exception('Error: order size too large; could not be executed. Extra amount is ' + str(vol_left) + '; t = ' + str(t))
    else:
        if add_info == False:
            return op * np.dot(bins, prices)
        else:
            return bins, op * (bins * prices), op * np.dot(bins, prices)

#   has volume as a function of all the data between start time and time t; this is very slow (~6 times slower than pnl_fast which has the same vol every time)

def pnl(pos, td, vol_func):
    #   this select function is very slow (200+ µs); considered also td.values[indices selected] so that I'm manipulating the np array instead but still slow
    vol_func_d_select = lambda t: td[:t.time()]

    def ff_order(t, op): 
        vol = vol_func(vol_func_d_select(t))
        return (fulfill_order(td, t, vol, op), vol)
    
    vec_ff_order = np.frompyfunc(ff_order, 2, 1)

    #   buys shares at times on pos; vectorised but still kinda slow (probably bc len(pos.index) < 100 in most cases, so negligible perf increase)
    t_buy = np.array(pos.index, dtype = object)
    t_sell = np.array(pos.index + timedelta(seconds = tdiff), dtype = object)

    loss = np.array(vec_ff_order(t_buy, -1)).transpose()
    profit = np.array(vec_ff_order(t_sell, 1)).transpose()

    total_ask_vol = np.sum(loss[:, 1])
    total_bid_vol = np.sum(profit[:, 1])

    avg_bid = np.sum(loss[:, 0])/total_ask_vol
    avg_ask = np.sum(profit[:, 0])/total_bid_vol
    
    return np.sum(profit[:, 0]), np.sum(loss[:, 0]), np.sum(profit[:, 0] + loss[:, 0]), avg_bid, avg_ask

def pnl_fast(pos, td, vol_scaling):    
    ff_order = lambda t, vl, op:  np.array([fulfill_order(td, t, vl, op), vl])
    #vec_ff_order = np.frompyfunc(ff_order, 3, 1)
    pos = pos_drop_zero(pos)

    #   buys shares at times on pos
    t_buy = np.array(pos.index, dtype = object)
    vols = (np.array(pos.values, dtype = float) * vol_scaling).astype(int).flatten()

    t_sell = np.array(pos.index + timedelta(seconds = tdiff), dtype = object)
    #   no idea why this is faster than vectorising
    loss = np.array([ff_order(t_buy[i], vols[i], -1) for i in range(len(vols))])
    profit = np.array([ff_order(t_sell[i], vols[i], 1) for i in range(len(vols))])

    total_ask_vol = np.sum(loss[:, 1])
    total_bid_vol = np.sum(profit[:, 1])

    avg_bid = np.sum(loss[:, 0])/total_ask_vol
    avg_ask = np.sum(profit[:, 0])/total_bid_vol
    
    return np.sum(profit[:, 0]), np.sum(loss[:, 0]), np.sum(profit[:, 0] + loss[:, 0]), avg_bid, avg_ask

def maketitle(x, td): 
    casenum = str(int(x[7:-16])) if x[0] == "l" else str(int(x[5:-16])) 
    return "buy case " + str(casenum) + "\n " + str(td.index[0])[:19] + " to " + str(str(td.index[-1])[:19])

#   graphs the figures like given
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
def get_rolling(trade_data, n=500, start = 0, log = False, drop_zero = False, interval = 1):
    data = trade_data[::interval]
    bid_price = data['bid_p1'][start:]

    if drop_zero == True:
        tick_price_diff = bid_price.values[1:] - bid_price.values[:-1]
        drop = np.flatnonzero(tick_price_diff == 0)
        bid_price = bid_price[drop]
    
    if log == True:
        bid_price = np.log(bid_price)

    mean = bid_price.rolling(int(n/interval)).mean()
    diff = bid_price-mean

    return pd.DataFrame(data= {'price': bid_price, 'rolling_mean':mean , 'diff': diff}, index = bid_price.index)

#   graphs price - price.rolling(n).mean
def plot_rolling(id, start = 0, interval = 1, n = 500):
    trade_data = to_df(td[id])
    pos_data = to_df(pos[id])
    data = get_rolling(trade_data, n, interval=interval)[start:]

    fig, ax = plt.subplots()
    ax.plot(data.index, data['diff'], label = 'price difference')
    ax.scatter(pos_data.index, np.zeros(len(pos_data.index))+1.2 * np.min(data['diff']), s= 1, c = 'g', marker= '^', label = 'buy times')
    ax.set(xmargin = 0, title = 'price - price.rolling(500).mean() for case {0}'.format(td[id][5:-16]))

    mean = np.mean(data['diff'])
    print(mean)
    ax.axhline(mean, c= 'r', label = 'mean')
    ax.legend()

def autocor(td_fname):
    returns = get_returns(td_fname)
    return acf(returns, nlags = 10)

#   graphs stuff for returns
def plot_returns(td_fname, start = 0, dlen = 0):
    returns = get_returns(td_fname, start, dlen)
    tp = returns.index

    fig, ax = plt.subplots()
    ax.set(xmargin = 0, xlabel = 'timestamp', ylabel = 'returns', title = "returns for {0}".format(td_fname[5:-16]))
    ax.plot(tp, returns, linewidth = 0.5)

def plot_autocor(td_fname, start = 0, dlen = 0):
    returns = get_returns(td_fname, start, dlen)
    plot_acf(returns, lags=15)
