from util_funcs import *
    
def dev_from_rolling_returns(td_fname, n , start = 0, dlen = 0):
    returns = get_returns(td_fname, start, dlen)
    t = start
    if start == 0:
        t = returns.index[0]

    returns = returns[t:]
    cummean = returns.rolling(n).mean()
    diff = returns-cummean
    return pd.DataFrame(data= {'return': returns, 'rolling_mean':cummean , 'diff': diff})

def plot_rolling(td_fname, n, ret = False, bins = 200, dlen = 0):
    data = dev_from_rolling_price(td_fname, n)['diff'].dropna() if ret == False else dev_from_rolling_returns(td_fname, n, dlen)['diff'].dropna()
    
    if dlen != 0:
        data = data[:dlen]

    x = np.linspace(-0.001, 0.001, 10000)
    
    b, loc, sc = sp.stats.gennorm.fit(data)
    y = sp.stats.gennorm.pdf(x, beta = b, loc = loc, scale = sc)
    xlim = [-5 * np.std(data), 5 * np.std(data)]
    if ret == False:
        xlim = [0.8 * np.min(data), -0.8 * np.min(data)]
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set(title = 'Histogram of difference between rolling mean of {0} and {0} for {1} \n rolling length = {2}, data length = {3}'.format('returns' if ret == True else 'price', td_fname[5:-16], n, len(data)), xlim = xlim, xlabel = 'Difference', ylabel = 'Density')
    counts, bins, bars = ax.hist(data, bins = bins, density = True)
    #ax.scatter(bars_mean, pred- counts, s= 0.5, c = 'r')
    ax.plot(x, y, label = r'Fitted sym. gen. norm. dist., $\mu \approx$ {0}, $\sigma \approx $ {1}, $ b \approx $ {2}'.format(np.round(loc, decimals = 8), np.round(sc, decimals = 8), np.round(b, decimals=8)))
    ax.legend(fontsize=8)

    skew = sp.stats.skew(data, bias = False)
    kurtosis = sp.stats.kurtosis(data, bias = False)
    #   creates real vs predicted data for goodness of fit and  performs 2 sample k-s test 
    bars_mean = (bins[:-1] + bins[1:])/2
    pred = sp.stats.gennorm.pdf(bars_mean, beta = b, loc = loc, scale = sc)
    fitp = sp.stats.ks_2samp(pred, counts).pvalue
    #   error as a percentage of mean abs err / area
    error = np.sum(np.abs(counts-pred))/np.sum(counts)
    
    #print("skew: {0}".format(skew))
    #print("kurtosis: {0}".format(kurtosis))
    #print("goodness of fit p val: {0}".format(sp.stats.ks_2samp(pred, counts).pvalue))
    #print("MAE/area: {0}".format(error))
    
    return pd.DataFrame(data = {'skew':skew, 'kurtosis':kurtosis, 'pvalue': fitp, 'error':error}, index = [data.index[0]])

def rolling_gen_norm_fit(td_fname, n):
    data = dev_from_rolling_price(td_fname, n)['diff']
    data = data[data != 0].dropna()

    b, loc, sc = sp.stats.gennorm.fit(data)
    return {'loc': loc, 'scale': sc, 'b': b}

def plot_fitted_param_changes(start = 0, end = len(td), param = 'scale'):
    gen_norm_fit_results_n10 = np.load("gen_norm_fit_results_n10.npy")
    gen_norm_fit_results_n100 = np.load("gen_norm_fit_results_n100.npy")
    gen_norm_fit_results_n500 = np.load("gen_norm_fit_results_n500.npy")

    fig, ax = plt.subplots(figsize= (20, 2))
    times = pd.DatetimeIndex(np.load("firstimes.npy", allow_pickle=True))
    fit_results_n100 = pd.DataFrame(data=gen_norm_fit_results_n100, index = times, columns = ['loc', 'scale', 'b'])

    fit_results_n100.sort_index(inplace=True)
    times_sorted = fit_results_n100.index

    ax.plot(fit_results_n100[param][start:end])
    ax.set(xmargin = 0, title = 'fitted distribution parameters per case; fitted param is {2}, \n start = {0}, end = {1}'.format(times_sorted[start], times_sorted[end], param))