from utils.util_funcs import * 


def momentum(id, time = 0, weight = 200, fixed_vol = False):
    trade_data = to_df(td[id])
    pos_data = to_df(pos[id])
    returns = get_returns(td[id])

    t = pos_data.index[0] if time == 0 else time

    select = trade_data[:t]['ask_p1']
    times = select.index
    diff = trade_data['ask_p1'][t] - select
    tdiff = (times - t).to_numpy(dtype = 'float64')
        
    #   normalises time diff to be between 0 and 100 
    tdiff = -tdiff/tdiff[0]* 100
    gradient = (-diff/tdiff).dropna()

    vols = trade_data['ask_tt_volume'][times].to_numpy()[:-1]
    if fixed_vol == True:
        vols = np.mean(vols)*np.ones(len(tdiff)-1)
        
    #   normalises weighted mean correctly
    norm_const = weight/10 *(1-np.exp(-1000/weight))

    #   returns momentum as sum of gradient * volume at time t * weighting
    #   where weighting exponentially decays; small weight = closer values weighted more
    return np.sum(gradient * vols * np.exp(-10 * np.abs(np.trim_zeros(tdiff)/weight)))/norm_const

def score(id, trade_data, pos_data, log = False, plot = False, show_all = False):
    returns = get_returns(trade_data)

    def mmtm(time, weight, fixed_vol = False):
        select = trade_data[:time]['ask_p1'] 

        times = select.index

        diff = trade_data['ask_p1'][time] - select
        tdiff = (times - time).to_numpy(dtype = 'float64')
        
        #   normalises time diff to be between 0 and 100 
        tdiff = -tdiff/tdiff[0]* 100
        gradient = (-diff/tdiff).dropna()

        vols = trade_data['ask_tt_volume'][times].to_numpy()[:-1]
        if fixed_vol == True:
            vols = np.mean(vols)*np.ones(len(tdiff)-1)
        
        #   normalises weighted mean correctly
        norm_const = weight/10 *(1-np.exp(-1000/weight))

        #   returns momentum as sum of gradient * volume at time t * weighting
        #   where weighting exponentially decays; small weight = closer values weighted more
        return np.sum(gradient * vols * np.exp(-10 * np.abs(np.trim_zeros(tdiff)/weight)))/norm_const
    
    pre_short_mom = np.array([mmtm(t, 200) for t in pos_data.index])
    pre_long_mom = np.array([mmtm(t, 100000) for t in pos_data.index])
    pre_short_mom_unweighted = np.array([mmtm(t, 200, fixed_vol=True) for t in pos_data.index])
    pre_long_mom_unweighted = np.array([mmtm(t, 100000, fixed_vol=True) for t in pos_data.index])
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(pos_data.index, pre_short_mom, label = 'short mom')
        ax.plot(pos_data.index, pre_long_mom, label = 'long mom')
        ax.plot(pos_data.index, pre_short_mom_unweighted, label = 'short unw mom')
        ax.plot(pos_data.index, pre_long_mom_unweighted, label = 'long unw mom')
        ax.set(xmargin = 0, title = 'Momentum score at buy times')
        ax.grid()
        ax.legend()

    mean_vol = 1/2*(np.mean(trade_data['ask_tt_volume']) + np.mean(trade_data['bid_tt_volume']))
    
    #   note that momentum approx. is linear with mean volume, so maybe consider normalising if we don't care about trade volume as much? 
    volatility = np.std(returns)
    if log == True:
        volatility = np.log(volatility * 10**5) * 100
    
    if show_all == True:
        #   calculate overall momentum score, weighting long term momentum more; essentially this is 3 parts long momentum 1 part short momentum (unweighted, since weighted short fluctuates like crazy) 
        mom_rms = 1/6 * np.sqrt(abs(np.sign(pre_short_mom)*pre_short_mom**2 + 2* np.sign(pre_long_mom)*pre_long_mom**2 + 2* np.sign(pre_long_mom_unweighted)*pre_long_mom_unweighted**2 + np.sign(pre_short_mom_unweighted) * pre_short_mom_unweighted**2))

        #   calculate overall score by taking rms of momentums & divide by volatility * 1000
        wghted_score = np.sign(pre_long_mom_unweighted) * mom_rms / (1000*volatility)

        n = len(pos_data.index)
        df = pd.DataFrame(data = np.array([wghted_score, mean_vol * np.ones(n), volatility * np.ones(n), pre_short_mom, pre_long_mom, pre_short_mom_unweighted, pre_long_mom_unweighted]).transpose(), columns = ['{0}score'.format("log " if log == True else ""), 'mean volume', '{0}volatility'.format("log " if log == True else ""), 'short mom', 'long mom', 'short unweighted mom', 'long unweighted mom'], index = pos_data.index)
        
        return df
    else: 
        short_mom = np.mean(pre_short_mom)
        long_mom = np.mean(pre_long_mom)
        unwghted_short_mom = np.mean(pre_short_mom_unweighted)
        unwghted_long_mom = np.mean(pre_long_mom_unweighted)

        mom_rms = 1/6 * np.sqrt(abs(np.sign(short_mom)*short_mom**2 + 2* np.sign(long_mom)*long_mom**2 + 2* np.sign(unwghted_long_mom)*unwghted_long_mom**2 + np.sign(unwghted_short_mom) * unwghted_short_mom**2))

        #   calculate overall score by taking rms of momentums & divide by volatility * 1000
        wghted_score = np.sign(unwghted_long_mom) * mom_rms / (1000*volatility)

        df = pd.DataFrame(data = np.array([wghted_score, mean_vol, volatility, short_mom, long_mom, unwghted_short_mom, unwghted_long_mom]).reshape(1, 7), columns = ['{0}score'.format("log " if log == True else ""), 'mean volume', '{0}volatility'.format("log " if log == True else ""), 'short mom', 'long mom', 'short unweighted mom', 'long unweighted mom'], index = [td[id][5:-16]])
        
        return df

def fast_mmtm(trade_data, pos_data, weight = 200, start = False, vol_scale = False):
    #   normalises weighted mean correctly
    norm_const = weight/10 *(1-np.exp(-1000/weight))
    vol_scaled = (trade_data['ask_v1']/trade_data['ask_v1'][0]).apply(lambda t: max(2, t))

    def score(t):
        select = trade_data[t-pd.Timedelta(minutes=18):t]
        asks = select['ask_p1']
        times = select.index
        diff = trade_data['ask_p1'][t] - asks
        tdiff = (times - t).to_numpy(dtype = 'float64')
            
        #   normalises time diff to be between 0 and 100 
        tdiff = -tdiff/tdiff[0]* 100
        gradient = (-diff/tdiff).dropna()

        if vol_scale == True:
            vol = vol_scaled[t-pd.Timedelta(minutes=18):t].iloc[:-1]
        else:
            vol = np.ones(len(gradient))

         #   returns momentum as sum of gradient * volume at time t * weighting
        #   where weighting exponentially decays; small weight = closer values weighted more
        return 10000 * np.sum(gradient * vol * np.exp(-10 * np.abs(np.trim_zeros(tdiff)/weight)))/norm_const
    
    if start == True:
        return score(pos_data.index[0])
    else:
        scores = pos_data.index.to_series().parallel_apply(score)
        return pd.Series(data = scores, index = pos_data.index)

all_scores = pd.read_csv(header + 'files/scored_cases.csv')
corr = np.corrcoef(all_scores['score'], all_scores['mean volume'])
#   correlation between them is approx -0.4, pretty sufficiently negatively correlated between score and mean vol

def pnl_by_thres(upper = -200.0, scoring = 'vol'):
    scoring_systems = {'log': 'log vol score', 'no_vol':'no vol score', 'vol': 'score'}
    score = scoring_systems[scoring]
    tradeid = all_scores[all_scores[score] <= upper]['id']
    res = np.zeros((len(pos), 5))
    print(len(tradeid))
    for i in tqdm(tradeid):
        j = int(str(i)[:-1]) if i != 0 else 0
        res[j, :] = pnl_fast(to_df(pos[j]), to_df(td[j]), 1000)
    
    pos_id = [x[5:-8] for x in pos]
    df_res = pd.DataFrame(data = res, index = pos_id, columns = ['profit', 'loss', 'net', 'avg bid', 'avg ask'])
    winrate = len(df_res[df_res['net'] > 0].index)/len(tradeid)
    df_res = df_res[df_res != 0].dropna()
    
    prof_given_w = np.mean(df_res[df_res['net']>0]['net'])
    loss_given_l = np.mean(df_res[df_res['net']<0]['net'])

    pnl = -prof_given_w/loss_given_l

    print(winrate, pnl)
    return df_res.dropna()

def plot_price_rolling_c(id, n):
    data = dev_from_rolling_price(td[id], n)

    price = data['price'][n-1:]
    rolling = data['rolling_mean'].dropna()
    gradient = rolling.values[1:]/rolling.values[:-1] * 10
    time = price.index

    fig, (ax, ax1) = plt.subplots(nrows = 2)
    ax.plot(time, price, label = 'price')
    ax.plot(time, rolling, label = 'rolling (n = {0})'.format(n))
    ax1.plot(time[1:], gradient, label = 'gradient')
    ax.set(xmargin = 0, title = 'price and rolling price')
    ax1.set(xmargin=0, title = 'gradient')
    ax.legend()
    ax1.legend()

def plot_price_rolling_diff(id):
    data = dev_from_rolling_price(td[id])
    price_diffs = data['price'].values[499:] - data['price'].values[:-499]
    means = data['diff'].dropna().rolling(500, center=True).mean().dropna()

    price_diffs_centred = price_diffs - price_diffs.mean()
    diff = data.dropna()['diff']
    diff_centred = diff - diff.mean()
    scale = price_diffs @ diff_centred / (diff_centred @ diff_centred)
    print(scale,  price_diffs.mean(),  1.5 * diff.mean())

    fig, ax = plt.subplots()
    ax.set(xmargin = 0)

    ax.plot(data.dropna().index, price_diffs)
    ax.plot(data.dropna().index, 1.5 * diff_centred, c = 'g')
    ax.plot(data.dropna().index, abs(price_diffs - 1.5 * diff_centred), c = 'r')
    ax.axhline(0, c = 'black')
    #ax.plot(data.dropna().index[250:-249], scale * means,)
    #ax.plot(data.dropna().index[250:-249], price_diffs[250:-249] - scale*means)

def vol_ratio_linreg(pre_vol, short_mom):
    x = 1/(1000 * pre_vol)
    y = -short_mom/10
    a = 0.128
    p = 1.174
    c = -0.106
    eq = (np.abs(a * x + (1-a) * y)**(p) + c * x)/100

    return 0.7604 + 20.534 * eq

def est_vol_ratio(trade_data, pos_data):
    data = get_returns(trade_data)
    start = pos_data.index[0] 
    pre_vol = np.std(data[:start][-6000:])
    short_mom = fast_mmtm(trade_data, pos_data, start = True)

    return vol_ratio_linreg(pre_vol, short_mom)