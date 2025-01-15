from ost import *

def expected_return_thres(id, thres = -3, M = 10, tdiff = 600, v = False, interval = 20):
    trade_data = to_df(td[id])
    pos_data = to_df(pos[id])

    N = int(tdiff/interval * 10)

    unscaled_score = fast_mmtm(trade_data, pos_data, weight = 2000)
    sc = get_returns(trade_data)[:unscaled_score.index[0]]
    score = unscaled_score/ (10**4 * np.std(sc))**2
    buy_pos = score[score <= thres]

    if len(buy_pos) == 0:
        print('no times less than threshold')
        return -999

    start = buy_pos.index[0]
    rolling = get_rolling(trade_data[:start])

    sp, ti = gen_OU_sample(trade_data, pos_data, start, N = 3* N, M = M, log = False, interval=interval)
    if np.min(sp) < -20:
        print('error with mle estimation')
        return -999
    
    price_paths = sp - sp[:, 0].reshape(M, 1) + 2.09294/(500/interval) * (np.cumsum(sp, axis = 1)) - 2.09294 * np.mean(rolling['diff'].dropna()) +  trade_data.at[start, 'bid_p1']
    if np.min(price_paths) < 0:
            print('error with mle estimation')
            return -999

    index = pd.date_range(start = start, periods = 3 * N, freq = '{0}00ms'.format(interval))

    df = pd.DataFrame(data = price_paths.transpose(), index = index)

    sell_pos_ind = (df.index).get_indexer(buy_pos.index + timedelta(seconds = tdiff), method = 'backfill')

    sell_prices = df.iloc[sell_pos_ind]
    ask_prices = trade_data['ask_p1'][buy_pos.index]

    if v == True:
        fig, ax = plt.subplots()
        ax.set(xmargin = 0)
        lt = len(trade_data['bid_p1'][buy_pos.index[0]::interval])
        ax.plot(trade_data['bid_p1'])
        ax.plot(df[0][:lt])
        ax1 = ax.twinx()
        ax1.plot(index[:lt], sp[0, :lt], c = 'g', alpha = 0.5)

        net = np.mean(sell_prices.values - ask_prices.values.reshape(len(sell_prices.values), 1), axis = 1) * np.abs(1000 * buy_pos.values)
        print(np.mean(net))
        return pd.Series(data =net ,index = buy_pos.index)
    else:
        net = np.mean(sell_prices.values - ask_prices.values.reshape(len(sell_prices.values), 1), axis = 1) * np.abs(1000 * buy_pos.values)

        return np.mean(net)

def payoff(id, interval, a, b, log = False):
    trade_data = to_df(td[id])
    pos_data = to_df(td[id])

    rolling = get_rolling(trade_data, interval = interval, log = log)['diff'].dropna()
    prices = trade_data['bid_p1'][::interval]

    if log == True:
        prices = np.log(prices)

    est_price = a * (rolling - rolling[0]) + b/(500/interval) * (np.cumsum(rolling) - (500/interval) * np.mean(rolling)) + prices[int(500/interval) -1]
    fig, ax = plt.subplots()
    #ax.plot(rolling)
    ax.plot(prices, label = 'real')
    ax.plot(est_price, label = 'est price from rolling')
    ax.legend()

    return np.sum((est_price - prices).dropna()**2)/np.sum(prices**2)

def partial_payoff(id, interval, k = 1.8, v = False, log = False):
    trade_data = to_df(td[id])
    pos_data = to_df(td[id])

    rolling = get_rolling(trade_data, interval = interval, log = log)['diff'].dropna()
    prices = trade_data['bid_p1'][::interval]
    if log == True:
        prices = np.log(prices)

    payoffs = prices[1:].values - prices[:-1]
    approx_payoffs = rolling.values[1:] - rolling[:-1] + k/(500/interval) * (rolling[:-1] - np.mean(rolling))

    err = 100 * np.sum((payoffs - approx_payoffs).dropna()**2)

    fig, ax = plt.subplots()
    #ax.plot(rolling)
    ax1 = ax.twinx()
    #ax1.plot(prices, c= 'g')
    ax.plot(payoffs, label = 'payoffs')
    ax.plot(approx_payoffs, label = 'est. payoffs')
    ax.legend()
    ax.set(xmargin = 0)
    plt.show()
    
    return err/np.sum(100 * approx_payoffs**2)

import optuna
def objective(trial):
    a = trial.suggest_float('ratio', 1.8, 2.2)
    b = trial.suggest_float('ratio2', 0.8, 1.2)
    error = [payoff(id, 10, a, b) for id in tqdm(range(10))]

    return np.sum(error)


exp_ret = np.load('exp_ret_unit.npy')
