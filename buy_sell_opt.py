from ost import * 

def graph_opt_sell(id, interval, N = 600):
    pos_data = to_df(pos[id])
    td_data = to_df(td[id])
    start = pos_data.index[int((len(pos_data.index)/2))]
    opt_stop = get_opt_stopping_time_batched(td_data, pos_data, start, N, 50, 2, 1, interval=interval, batches=[15, 2])
    ends = [start + timedelta(seconds= 0.1 * int(interval * st)) for st in opt_stop]
    print(opt_stop)

    fig, ax = plt.subplots()
    ax.set(xmargin = 0, title = maketitle(td[id], td_data), xlabel = "timestamp")
    ax.plot([start], td_data["ask_p1"].at_time(start) * (0.9987), marker = "^", color = "red", markersize = '4', alpha = 0.8)
    ax.plot(ends, td_data["bid_p1"].loc[ends] * (1.0013), marker = "v", color = "green", markersize = '4', alpha = 0.8)
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

#   now takes 3 buy times 
def buy_thres(trade_data, pos_data, thres = -3, interval = 20, N = 600, v = False, p = False, batches=[12, 3]):
    unscaled_score = fast_mmtm(trade_data, pos_data, weight = 2000)
    sc = get_returns(trade_data)[:unscaled_score.index[0]]
    score = unscaled_score/ (10**4 * np.std(sc))**2
    buy_pos = score[score <= thres]
    buy_pos_len = len(buy_pos)

    if len(buy_pos) == 0:
        print('no times in pos that are below threshold. no trades performed')
        return -np.ones(4 + batches[1])

    stopping_times = get_opt_stopping_time_batched(trade_data, pos_data, buy_pos.index[-1], N, 50, 2, 1, v = v, interval=interval, batches=batches)

    sell_pos_indexes = [buy_pos.index + pd.Timedelta(buy_pos.index[-1] - buy_pos.index[0]) + pd.Timedelta(timedelta(seconds = int(0.1 * interval * st))) for st in stopping_times]

    n_batches = len(stopping_times)
    
    buy = np.zeros((buy_pos_len, 2))
    sell = np.zeros((buy_pos_len * n_batches, 2))
    vol = lambda x: int(min(4000, 100 * abs(x)))

    for i in range(buy_pos_len):
        buy_vol = vol(buy_pos[i]) * n_batches
        sell_vol = vol(buy_pos[i])

        buy[i, :] = [fulfill_order(trade_data, buy_pos.index[i], buy_vol, -1), buy_vol]

        for j in range(n_batches):
            sell[i + buy_pos_len * j, :] = [fulfill_order(trade_data, buy_pos.index[i] + pd.Timedelta(buy_pos.index[-1] - buy_pos.index[0]) + timedelta(seconds=0.1 * interval * int(stopping_times[j])),sell_vol, 1), sell_vol]
    
    df_buy = pd.DataFrame(data = buy, index = buy_pos.index, columns = ['change', 'vol'])
    df_sell = pd.concat([(pd.DataFrame(data = sell[i * buy_pos_len: (i+1) * buy_pos_len], index = sell_pos_indexes[i], columns = ['change', 'vol'])) for i in range(n_batches)])
    
    if p == True:
        fig, ax = plt.subplots()
        start = df_buy.index[0] - timedelta(minutes=30)
        end = df_sell.index[0] + timedelta(minutes=30)
        ax.plot(trade_data['bid_p1'][start:end])
        ax.scatter(df_buy.index, 0.999 * trade_data['ask_p1'].loc[df_buy.index], marker = '^', c = 'r',alpha = 0.2)
        ax.scatter(df_sell.index, 1.001 * trade_data['bid_p1'].loc[df_sell.index], marker = 'v', c = 'g', alpha = 0.2)
        ax.set(xmargin = 0)
    
    if v == True:
        print(stopping_times)
        print(np.sum(sell[:, 0]) + np.sum(buy[:, 0]) )
        return pd.concat((df_buy, df_sell))
    else:
        return np.hstack(([np.sum(buy[:, 0]), np.sum(sell[:, 0]), np.sum(sell[:, 0]) + np.sum(buy[:, 0]), np.sum(buy[:, 1])], stopping_times))

def get_all_res(fname):
    file = header + 'files/{0}.npy'.format(fname)
    result = np.load(file) if os.path.isfile(file) else np.zeros((len(td), 7))
    ids = tqdm(range(np.argmin(result[:, 1] != 0), len(pos)))

    for id in ids:
        t = to_df(td[id])
        p = to_df(pos[id])
        try:
            result[id, :] = np.array(buy_thres(t, p, thres=-1, N = 450))
            np.save(file, result)
        except:
            print('error occured: id num = {0}'.format(id))
            result[id, :] = -np.ones(7)
            np.save(file, result)

def bt(id, interval):
    t = to_df(td[id])
    p = to_df(pos[id])

    return buy_thres(t, p, interval=interval, p = True)

def st(id, interval, N = 600):
    t = to_df(td[id])
    p = to_df(pos[id])
    
    stopping_time = get_opt_stopping_time_batched(t, p, p.index[0], N, 50, 2, 1, v = True, interval=interval, batches=[15, 2])

#get_all_res('opt_stopping_thres_l')

def get_sell_times(volume, ref_time, freq):
    ref_time = pd.Timestamp(ref_time)
    window = min(50, int(volume/7000))
    sell_times = pd.date_range(ref_time - window * pd.Timedelta(freq), ref_time + window * pd.Timedelta(freq), freq = freq, inclusive = 'right')

    length = len(sell_times)
    vol = volume/length
    return pd.Series(data = vol *np.ones(length), index = sell_times)

#   now takes 3 buy times 
def buy_thres_fixed(trade_data, pos_data, thres = -3, interval = 20, N = 600, v = False, p = False, batches=[12, 3]):
    unscaled_score = fast_mmtm(trade_data, pos_data, weight = 2000)
    sc = get_returns(trade_data)[:unscaled_score.index[0]]
    score = unscaled_score/ (10**4 * np.std(sc))**2
    buy_pos = score[score <= thres]
    buy_pos_len = len(buy_pos)

    if len(buy_pos) == 0:
        print('no times in pos that are below threshold. no trades performed')
        return -np.ones(4 + batches[1])

    stopping_times = np.round(get_opt_stopping_time_batched(trade_data, pos_data, buy_pos.index[-1], N, 50, 2, 1, v = v, interval=interval, batches=batches), decimals = 1)
    n_batches = len(stopping_times)
    
    buy = np.zeros((buy_pos_len, 2))
    vol = lambda x: int(min(4000, 100 * abs(x)))

    for i in range(buy_pos_len):
        buy_vol = vol(buy_pos[i]) * n_batches
        buy[i, :] = [fulfill_order(trade_data, buy_pos.index[i], buy_vol, -1), buy_vol]

    total_vol = np.sum(buy[:, 1])
    ref_times = buy_pos.index[-1] + stopping_times * interval * pd.Timedelta(milliseconds=100)
    sell_times = pd.concat([get_sell_times(total_vol/n_batches, rt, '0.5s') for rt in ref_times])
    n_sell = len(sell_times)
    sell = np.zeros((n_sell, 2))

    for i in range(n_sell):
        sell[i, :] = [fulfill_order(trade_data, sell_times.index[i], sell_times.iloc[i], 1), sell_times.iloc[i]]

    df_buy = pd.DataFrame(data = buy, index = buy_pos.index, columns = ['change', 'vol'])
    df_sell = pd.DataFrame(data = sell, index = sell_times.index, columns = ['change', 'vol'])
    
    if p == True:
        fig, ax = plt.subplots()
        start = df_buy.index[0] - timedelta(minutes=30)
        end = df_sell.index[0] + timedelta(minutes=30)
        ax.plot(trade_data['bid_p1'][start:end])
        ax.scatter(df_buy.index, 0.999 * trade_data['ask_p1'].loc[df_buy.index], marker = '^', c = 'r',alpha = 0.2)
        ax.scatter(df_sell.index, 1.001 * trade_data['bid_p1'].loc[df_sell.index], marker = 'v', c = 'g', alpha = 0.2)
        ax.set(xmargin = 0)
    
    if v == True:
        print(stopping_times)
        print(np.sum(sell[:, 0]) + np.sum(buy[:, 0]) )
        return pd.concat((df_buy, df_sell))
    else:
        return np.hstack(([np.sum(buy[:, 0]), np.sum(sell[:, 0]), np.sum(sell[:, 0]) + np.sum(buy[:, 0]), np.sum(buy[:, 1])], stopping_times * interval/10))
