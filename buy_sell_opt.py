from analysis import *
from optimal_stopping_times import get_opt_stopping_time_batched
from multiprocess import Pool

def graph_opt_sell(id, interval, N = 600):
    pos_data = to_df(pos[id])
    td_data = to_df(td[id])
    start = pos_data.index[int((len(pos_data.index)/2))]
    opt_stop = get_opt_stopping_time_batched(id, start, N, 50, 2, 10, interval=interval, batches=[10, 2])
    end = start + timedelta(seconds= 0.1 * int(interval * opt_stop))
    print(opt_stop)

    fig, ax = plt.subplots()
    ax.set(xmargin = 0, title = maketitle(td[id], td_data), xlabel = "timestamp")
    ax.plot([start], td_data["ask_p1"].at_time(start) * (0.9987), marker = "^", color = "red", markersize = '4', alpha = 0.8)
    ax.plot([end], td_data["bid_p1"].at_time(end) * (1.0013), marker = "v", color = "green", markersize = '4', alpha = 0.8)
    ax.plot(td_data.index, td_data["bid_p1"])
    print(td_data["ask_p1"].at_time(end).values - td_data["bid_p1"].at_time(start).values)
    
def score_plot(id, weight = 2000):
    fig, ax = plt.subplots()
    p = to_df(pos[id])
    tdd = to_df(td[id])
    ret = get_returns(td[id])
    print(np.std(ret))
    ax.plot(tdd[p.index[0]-timedelta(minutes=2):p.index[-1] + timedelta(minutes=2)]['bid_p1'])
    ax.set(xmargin = 0)
    ax1 = ax.twinx()

    long = fast_mmtm(id, weight = weight) / (10**4 * np.std(ret))**2
    ax1.plot(long, c='b')
    ax1.axhline(-4.5, c = 'black')

def buy_thres(id, thres = -3, interval = 20, N = 600, v = False, p = False):
    ret = get_returns(td[id])
    score = fast_mmtm(id, weight = 2000) / (10**4 * np.std(ret))**2
    buy_pos = score[score <= thres]
    trade_data = to_df(td[id])

    if len(buy_pos) == 0:
        print('no times in pos that are below threshold. no trades performed')
        return -np.ones(5)
    
    stopping_time = get_opt_stopping_time_batched(id, buy_pos.index[0], N, 50, 2, 10, v = v, interval=interval, batches=[10, 2])

    sell_pos_ind = buy_pos.index + timedelta(seconds=0.1 * interval * int(stopping_time))

    buy = np.zeros((len(buy_pos), 2))
    sell = np.zeros((len(buy_pos), 2))

    for i in range(len(buy_pos)):
        buy[i, :] = [fulfill_order(trade_data, buy_pos.index[i], int(100 * abs(buy_pos[i])), -1), int(100 * abs(buy_pos[i]))]
        sell[i, :] = [fulfill_order(trade_data, buy_pos.index[i] + timedelta(seconds=0.1 * interval * int(stopping_time)), int(100 * abs(buy_pos[i])), 1), int(100 * abs(buy_pos[i]))]

    df_buy = pd.DataFrame(data = buy, index = buy_pos.index, columns = ['change', 'vol'])
    df_sell = pd.DataFrame(data = sell, index = sell_pos_ind, columns = ['change', 'vol'])
    if p == True:
        plt.plot(trade_data['bid_p1'])
        plt.plot(df_buy.index, np.mean(trade_data['bid_p1']) * np.ones(len(buy_pos)), marker = '^', c = 'r',alpha = 0.2)
        plt.plot(df_sell.index, np.mean(trade_data['bid_p1']) * np.ones(len(buy_pos)), marker = 'v', c = 'g', alpha = 0.2)
    
    if v == True:
        print(stopping_time)
        print(np.sum(sell[:, 0]) + np.sum(buy[:, 0]) )
        return pd.concat((df_buy, df_sell))
    else:
        return np.array([np.sum(buy[:, 0]), np.sum(sell[:, 0]), np.sum(sell[:, 0]) + np.sum(buy[:, 0]), np.sum(buy[:, 1]), stopping_time])

def get_res(i):
    print(i)
    return buy_thres(i, v = False)
   
def get_all_res():
    result = np.load('files/opt_stopping.npy')
    ids = tqdm(range(np.argmin(result[:, 1] != 0), 523))
    for id in ids:
        result[id, :] = np.array(buy_thres(id))
        np.save('files/opt_stopping.npy', result)

get_all_res()