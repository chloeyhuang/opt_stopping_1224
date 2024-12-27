from analysis import *

#   stuff for vol modelling
scaling = lambda x: np.abs(np.tanh((x-1)))
vol_metrics = pd.read_csv('files/vol_metrics.csv')

mle_all = pd.read_csv('files/mle_OU_ests.csv')
mle_an = pd.read_csv('files/mle_OU_stats.csv')

"""
#   this is just how the metrics were produced; I get this is technically using future data but assuming this is the training set I can then try on more data
lm_vol = sp.stats.linregress(vol_metrics['vol index'], mle_an['r_sigma'])
vol_index_regressed = lm_vol.intercept + lm_vol.slope * vol_index

vol_index_r_adjusted =(1-scaling(buy_vol_ratio)) * vol_index_regressed + scaling(buy_vol_ratio) * buy_vol_ratio
"""

def OU_process(dt, X0, N, theta, mu, sigma):
    X = np.zeros(N)
    X[0] = X0

    # Generate the OU process
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    return X

#   some constants for mle estimation
def beta1(X):
    n = len(X)-1
    top = 1/n * np.sum(X[1:] * X[:-1]) - 1/(n**2) * np.sum(X[1:]) * np.sum(X[:-1])
    bottom = 1/n * np.sum(X[:-1]**2) - 1/(n**2) * (np.sum(X[1:]))**2
    return top/bottom

def beta2(X):
    n = len(X)-1
    return 1/n * np.sum(X[1:] - beta1(X)*X[:-1])/(1-beta1(X))

def beta3(X):
    b1 = beta1(X)
    b2 = beta2(X)
    n = len(X)-1

    inside = lambda x1, x0: (x1 - b1*x0 - b2 * (1-b1))**2
    inside = np.frompyfunc(inside, 2, 1)
    return 1/n * np.sum(inside(X[1:], X[:-1]))

#   mle / bootstrap estimation
def mle_est(X, dt, est_mu = False):
    theta = -1/dt * np.log(beta1(X)) #  speed of mean reversion
    sigma = np.sqrt(2*theta*beta3(X)/(1-beta1(X)**2))  #    volatility
    mu = 0 if est_mu == False else beta2(X) #   mean
    return theta, mu, sigma

def bs_est(X, dt, num = 10, est_mu = False):
    theta_mle, mu_mle, sigma_mle = mle_est(X, dt, est_mu)
    N = len(X)
    X0 = X[0]
    theta_bs = [theta_mle]
    mu_bs = [mu_mle]
    sigma_bs = [sigma_mle]

    for i in tqdm(range(num)):
        Y = OU_process(dt, X0, N, theta_bs[-1], mu_bs[-1], sigma_bs[-1])
        theta, mu, sigma = mle_est(Y, dt, est_mu)
        theta_bs.append(theta)
        mu_bs.append(mu)
        sigma_bs.append(sigma)

    theta_mean = np.mean(theta_bs)
    mu_mean = np.mean(mu_bs)
    sigma_mean = np.mean(sigma_bs)
    
    # bias correction
    theta_b = 2*theta_mle - theta_mean 
    mu_b = 2*mu_mle - mu_mean
    sigma_b = 2*sigma_mle - sigma_mean
    return theta_b, mu_b, sigma_b

def plot_OU(dt, X0, N, theta, mu, sigma):
    Y = OU_process(dt, X0, N, theta, mu, sigma)
    theta_est, mu_est, sigma_est = bs_est(Y, est_mu=True)
    theta_mle, mu_mle, sigma_mle = mle_est(Y, dt, est_mu=True)
    print(theta, mu, sigma)
    print("bootstrapped")
    print(theta_est, mu_est, sigma_est)
    print("mle")
    print(theta_mle, mu_mle, sigma_mle)

    # Plot the result
    fig, ax = plt.subplots()
    t = np.arange(0, N*dt, dt)
    ax.plot(t, Y)
    ax.axhline(y = mu, c = 'g')
    ax.set(title = r"$\mu = {0}, \theta = {1}, \sigma = {2}$".format(mu, theta, sigma), xlabel = "time", ylabel = 'X(t)', xmargin = 0)

def plot_OU_est(dt, X0, start, N, theta, mu, sigma, id):
    data = get_rolling(id)['diff'].dropna()
    X0 = data[start]
    Y = OU_process(dt, X0, N, theta, mu, sigma)
    # Plot the result
    fig, ax = plt.subplots()
    ax.plot(data.index, data, label = 'original')
    ax.plot(data[start:].index, Y, label = 'simulated')
    ax.set(title = "OU MLE estimation vs real data, case {0}0".format(id), xlabel = "time", ylabel = 'X(t)', xmargin = 0)
    ax.legend()

def gen_cdf(X, dt, bs_num = 0, stepsize = 0.01, v = False):
    n = len(X)

    theta, mu, sigma = bs_est(X, dt, bs_num) if bs_num >0 else mle_est(X, dt)

    emp_dist = 1/np.sqrt(dt * sigma**2) * (X[1:] - (1-theta * dt) * X[:-1]- dt * theta * mu) 

    if v == True:
        emp_dist_filtered = np.extract(np.abs(emp_dist) >= 0.05, emp_dist)

        print(n, "| ", theta, sigma)
        print("prop small: ", np.count_nonzero(np.abs(emp_dist)<=0.05)/len(emp_dist))

    smoothed_emp_cdf = lambda X: lambda x: 1/len(X) * np.sum(sp.stats.norm.cdf((100/(1+100*x**2)+5) * (x-X)))
    #h scaling = lambda x: 100/(1+100*x**2)+5 
    
    smoothed_emp_cdf = np.frompyfunc(smoothed_emp_cdf(emp_dist),1,1)
    t = np.arange(-5, 5, stepsize)
    return t, smoothed_emp_cdf(t), theta, sigma

def OU_process_mod(dt, X0, N, theta, mu, sigma, cdf_ran, cdf):
    X = np.zeros(N)
    X[0] = X0
    # pre generate the random numbers 
    r = np.random.uniform(0, 1, size=N)
    
    # Generate the OU process
    for t in range(1, N):    
        rnum = cdf_ran[np.argmin(cdf < r[t])]
        dW = np.sqrt(dt) * rnum    
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW

    return X

def OU_est(id, log = True, bs_num = 0, r_corrected = True, interval = 1):    
    data = get_rolling(id, log = log, interval = interval)['diff'].dropna()
    start = to_df(pos[id]).index[0]

    pre_train = data[:start]
    N = len(data) - len(pre_train) + 1
    dt = 0.1 * interval

    vol_ratio = mle_an['r_sigma'][id]
    pred_vol_ratio = vol_metrics['r_sigma pred.'][id]
    print('vol ratio: ', np.round(vol_ratio, decimals=4), ' | ', 'pred vol ratio: ', np.round(pred_vol_ratio, decimals=4))
    
    cdf_ran, cdf, theta, sigma = gen_cdf(pre_train.values, dt, bs_num=bs_num)
    pred = OU_process_mod(dt, pre_train[-1], N, theta, 0, sigma, cdf_ran, cdf)
    pred_ratio_corrected = OU_process_mod(dt, pre_train[-1], N, theta, 0, pred_vol_ratio * sigma, cdf_ran, cdf)
    pred_index = data[start:].index
    if len(pred) != len(pred_index):
        pred_index = data[start:].reindex(pd.date_range(start = start - timedelta(milliseconds=100*interval),end = pred_index[-1], freq='{0}00ms'.format(interval))).index

    fig, ax = plt.subplots()
    ax.plot(data.index, data, label = 'true')
    ax.plot(pred_index, pred, label = 'predicted')
    ax.plot(pred_index, pred_ratio_corrected, label = 'predicted (vol corrected)')
    ax.set(xmargin = 0)
    ax.legend()

def gen_OU_sample(id, time, log = True, N = 6000, M = 1, emp = True, interval = 1):
    data = get_rolling(id, log = log, interval=interval)['diff'].dropna()
    pos_data = to_df(pos[id])
    #start = pos_data.index[int((len(pos_data.index)/2))]
    train = data[:time]
    dt = 0.1 * interval

    #vol_ratio = mle_an['r_sigma'][id]
    pred_vol_ratio = est_vol_ratio(id)
    #if v== True:
        #print('vol ratio: ', np.round(vol_ratio, decimals=4), ' | ', 'pred vol ratio: ', np.round(pred_vol_ratio, decimals=4))
    
    cdf_ran, cdf, theta, sigma = gen_cdf(train.values, dt, bs_num=0)
    paths = np.zeros((M, N))

    if emp == True:
        paths = np.apply_along_axis(lambda x: OU_process_mod(dt, train[-1], N, theta, 0, pred_vol_ratio * sigma, cdf_ran, cdf), -1, paths)
    else:
        paths = np.apply_along_axis(lambda x: OU_process(dt, train[-1], N, theta, 0, pred_vol_ratio* sigma), -1, paths)
    
    time_index = np.arange(0, N*dt, dt)
    return paths, time_index

