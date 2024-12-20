from util_funcs import *

acfs = np.load("files/acfs_all_samples.npy")
sk_kt_all = pd.DataFrame(data = np.load("files/skew_kurtosis_all_samples.npy"), columns=['prop. nonzero', 'skew', 'kurtosis'], index = [i[5:-16] for i in td])

def plot_hist_returns(td_fname, nozero = True, n = 1, dlen = 0):
    returns = get_returns(td_fname, dlen=dlen).rolling(n).mean().dropna()
    total_len = len(returns)
    if nozero == True:
        returns = returns[returns != 0].rolling(n).mean().dropna()
        if n == 1:
            print("nonzero proportion: {0}".format(len(returns)/total_len))
       
    print("skew: {0}".format(sp.stats.skew(returns, bias = False)))
    print("kurtosis: {0}".format(sp.stats.kurtosis(returns, bias = False)))
    x = np.linspace(-0.0015, 0.0015, 10000)
    fitmean, fitstd = sp.stats.norm.fit(returns)
    yfit = sp.stats.norm.pdf(x, loc = fitmean, scale = fitstd)

    fig, ax = plt.subplots()
    ax.set(xlim = [-6 * np.std(returns), 6 * np.std(returns)], xlabel = 'Return', ylabel = "Density", title = "Histogram of returns for case {0}, n = {1}".format(td_fname[5:-16], n))
    ax.hist(returns, bins = 500, density = True)
    ax.plot(x, yfit, label =  r'fitted normal dist, $\mu \approx$ {0}, $\sigma \approx $ {1}'.format(np.round(fitmean, decimals = 9), np.round(fitstd, decimals = 9)))
    ax.legend()

def compute_skew_kurtosis(td_fname, nozero = True):
    returns = get_returns(td_fname)
    total_len = len(returns)
    if nozero == True:
        returns = returns[returns != 0]
    
    return np.count_nonzero(returns)/total_len, sp.stats.skew(returns, bias = False), sp.stats.kurtosis(returns, bias = False)

def plot_all_returns():
    all_returns =  pd.concat([get_returns(name) for name in td])
    print("skew: {0}".format(sp.stats.skew(all_returns, bias = False)))
    print("kurtosis: {0}".format(sp.stats.kurtosis(all_returns, bias = False)))
    #a, mean, sc = sp.stats.skewnorm.fit(all_returns)
    #print(a, mean, sc)
    #x = np.linspace(-0.0015, 0.0015, 150)
    #y = sp.stats.skewnorm.pdf(x, a = a, loc = mean, scale = sc)

    fig, ax = plt.subplots()
    ax.set(xlim = [-0.001, 0.001], xlabel = 'Return', ylabel = "Density", title = "Histogram of returns for all data")

    return ax.hist(all_returns[all_returns != 0 ], bins = 10000, density = True)

def norm_fit_rolling(td_fname, start_time, hist_length):
    all_data = dev_from_rolling_price(td_fname, 100)['diff']
    all_data = all_data[all_data != 0].dropna()
    data = all_data[pd.Timestamp(start_time):][:hist_length]

    b, loc, sc = sp.stats.gennorm.fit(data)
    return {'loc': loc, 'scale': sc, 'b': b}

def plot_local_fit_changes(td_fname, hist_length):
    all_data = dev_from_rolling_price(td_fname, 100)['diff']
    all_data = all_data[all_data != 0].dropna()

    b_vals = []
    loc_vals = []
    scale_vals = []
    skews = []

    times = all_data.index
    fit_times = times[:-hist_length][0::50]
    print(len(fit_times))
    for t in fit_times:
        data = all_data[pd.Timestamp(t):][:hist_length]
        b, loc, sc = sp.stats.gennorm.fit(data)

        b_vals.append(b)
        loc_vals.append(loc)
        scale_vals.append(sc)
        skews.append(sp.stats.skew(data, bias = False))

    fig, (axb, axloc, axsc, axsk) = plt.subplots(nrows=4, ncols=1, figsize=(10,6), sharex=True)
    axb.plot(fit_times, b_vals)
    axloc.plot(fit_times, loc_vals)
    axsc.plot(fit_times, scale_vals)
    axsk.plot(fit_times, skews)

    axb.set(xlabel = 'Time', ylabel = 'b', xmargin = 0, title = 'Fitted b, mean, std, and skew of distribution of price - price.rolling(100).mean(), rolling data length {0} for case {1}'.format(hist_length, td_fname[7:-16]))
    axsc.set(xlabel = 'Time', ylabel = 'std', xmargin = 0)
    axloc.set(xlabel = 'Time', ylabel = 'mean', xmargin = 0)
    axsk.set(xlabel = 'Time', ylabel = 'skew', xmargin = 0)
    fig.tight_layout()

fit_all = pd.read_csv("files/gennormfit_info.csv", index_col = 0)