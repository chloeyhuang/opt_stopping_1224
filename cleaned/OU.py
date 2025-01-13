from util_funcs import *

#   normal OU process with normal dW
def OU_process(dt, X0, N, theta, mu, sigma):
    X = np.zeros(N)
    X[0] = X0
    r = np.random.normal(0, 1, size = N)

    for t in range(1, N):
        dW = np.sqrt(dt) * r[t-1]
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    return X

#   OU process with non normal dW (custom distribution via cdf)
def OU_process_mod(dt, X0, N, theta, mu, sigma, cdf_ran, cdf):
    X = np.zeros(N)
    X[0] = X0

    # pre generate the random numbers for speed
    r = np.random.uniform(0, 1, size=N)
    
    # Generate the OU process
    for t in range(1, N):    
        rnum = cdf_ran[np.argmin(cdf < r[t])]
        dW = np.sqrt(dt) * rnum    
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

#   mle estimation
def mle_est(X, dt, est_mu = False):
    theta = -1/dt * np.log(beta1(X)) #  speed of mean reversion
    sigma = np.sqrt(2*theta*beta3(X)/(1-beta1(X)**2))  #    volatility
    mu = 0 if est_mu == False else beta2(X) #   mean
    return theta, mu, sigma

#   bootstrap estimation on mle est - note that this doesn't really work due to convergence issues
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

#   generates empirical distribution of dW from price - rolling mean timeseries, assuming that its approx mean reverting and OU process - esque
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

#   generates OU sample from trade data; pos data used for buy times for volatility ratio prediction
def gen_OU_sample(trade_data, time, log = False, N = 6000, M = 1, emp = True, interval = 1):
    data = get_rolling(trade_data, log = log, interval=interval)['diff'].dropna()
    train = data[:time]
    dt = 0.1 * interval

    pred_vol_ratio = est_vol_ratio(trade_data, time)
    
    cdf_ran, cdf, theta, sigma = gen_cdf(train.values, dt, bs_num=0)
    paths = np.zeros((M, N))

    if emp == True:
        paths = np.apply_along_axis(lambda x: OU_process_mod(dt, train[-1], N, theta, 0, pred_vol_ratio * sigma, cdf_ran, cdf), -1, paths)
    else:
        paths = np.apply_along_axis(lambda x: OU_process(dt, train[-1], N, theta, 0, pred_vol_ratio* sigma), -1, paths)
    
    time_index = np.arange(0, N*dt, dt)
    return paths, time_index

