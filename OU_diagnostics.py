from OU import * 

def OU_diff(id, plot = True, est_mu = False, log=False):
    data = get_rolling(id, log = log)['diff'].dropna()
    start = to_df(pos[id]).index[0]

    pre_train = data[:start]
    post_train = data[start:]
    N = len(post_train)
    dt = 0.1

    theta_0, mu_0, sigma_0 = mle_est(pre_train.values, dt, est_mu)
    theta_1, mu_1, sigma_1 = mle_est(post_train.values, dt, est_mu)
    r_theta, r_mu, r_sigma = theta_1/theta_0, mu_1/mu_0 if mu_0 != 0 else 0 , sigma_1/sigma_0
    
    if plot == True:
        print("pre:", theta_0, mu_0, sigma_0)
        print("post:", theta_1, mu_1, sigma_1)
        print("ratio:", r_theta, r_mu, r_sigma)
        X0 = data[start]

        Y0 = OU_process(dt, X0, N, theta_0, mu_0, sigma_0)
        Y1 = OU_process(dt, X0, N, theta_1, mu_1, sigma_1)

        fig, ax = plt.subplots()
        ax.plot(data.index, data, label = 'original')
        ax.plot(data[start:].index, Y0, label = 'simulated (pre buy)')
        ax.plot(data[start:].index, Y1, label = 'simulated (post buy)')
        ax.set(title = "OU MLE estimation vs real data, case {0}0".format(id), xlabel = "time", ylabel = 'X(t)', xmargin = 0)
        ax.legend()
    else:
        cols = ['theta_0', 'theta_1', 'r_theta', 'mu_0', 'mu_1', 'r_mu', 'sigma_0', 'sigma_1', 'r_sigma']
        vals = np.array([theta_0, theta_1, r_theta, mu_0, mu_1, r_mu, sigma_0, sigma_1, r_sigma]).reshape(1, 9)
        caseid = id*10

        return pd.DataFrame(data=vals, columns = cols, index = [caseid])

def gof(id, v = False, alpha = 0.05, bs_num = 1, log = True, bins = 800, y_upper = 1.0, x_ran = 5, filter = False):
    data = get_rolling(id, log = log)['diff'].dropna()
    start = pos_drop_zero(to_df(pos[id])).index[0]

    X = data[:start].values
    n = len(X)
    dt = 0.1 

    theta, mu, sigma = bs_est(X, dt, bs_num) if bs_num >0 else mle_est(X, dt)
    print(n, "| ", theta, sigma)
    emp_dist = 1/np.sqrt(0.1 * sigma**2) * (X[1:] - (1-theta * 0.1) * X[:-1]- 0.1 * theta * mu) 
    emp_dist_filtered = np.extract(np.abs(emp_dist) >= 0.05, emp_dist)
    print("prop small: ", 1- len(emp_dist_filtered)/len(emp_dist))

    def smoothed_emp_cdf(h, X):
        n = len(X)
        h_scale = lambda x: 100/(1+100*x**2)+5
        return lambda x: 1/n * np.sum(sp.stats.norm.cdf(h_scale(x) * (x-X)))
    
    smoothed_emp_cdf = np.frompyfunc(smoothed_emp_cdf(3, emp_dist),1,1)
    t = np.linspace(-5, 5, 1000)
    smoothed_pdf = 1/0.001 * (smoothed_emp_cdf(t+0.001) - smoothed_emp_cdf(t))
    #print(np.sum(smoothed_pdf)/100) makes sure that the area under pdf is approx 1

    fig, ax = plt.subplots()
    if filter == True:
        ax.hist(emp_dist_filtered, density = True, bins = bins, label = 'Empirical Distribution')
    else:
        ax.hist(emp_dist, density = True, bins = bins, label = 'Empirical Distribution')
    #ax.plot(t, smoothed_emp_cdf(t))
    ax.plot(t, smoothed_pdf, label = 'smoothed emp. dist.')
    ax.plot(t, (sp.stats.norm.pdf(t)), label = r'$Z \sim N(0, 1)$')
    ax.set(xlim = [-x_ran, x_ran], ylim = [-0, y_upper], title = 'Empirical dist. of ' + r"$\frac{1}{\sqrt{dt \sigma^2}}(X_t - (1-\theta dt)X_{t-1})$" + ' for case {0}0'.format(id))
    ax.legend()
    print(sp.stats.norm.fit(emp_dist), sp.stats.skew(emp_dist))
    test_stat = 1/np.sqrt(n * dt * sigma**2) * (X[-1]- X[0] + dt * theta * np.sum(X[:-1]))
    
    cf = sp.stats.norm.interval(1-alpha)
    if test_stat > cf[0] and test_stat < cf[1]:
        if v == True:
            print("the test statistic is {0} which is within the 5% conf. int. of {1}".format(test_stat, cf))
        return True
    else:
        if v == True:
            print("the test statistic is {0} which is not within the 5% conf. int. of {1}".format(test_stat, cf))
        return False

def mle_error_test(N, theta, sigma, bs_num = 10):
    X = OU_process(0.1, 0, N, theta, 0, sigma)
    dt = 0.1

    theta_est, mu_est, sigma_est = bs_est(X, dt, bs_num) if bs_num >0 else mle_est(X, dt)
    print("true")
    print(theta, sigma)
    
    print('est')
    print(theta_est, sigma_est)
    emp_dist = 1/np.sqrt(0.1 * sigma_est**2) * (X[1:] - (1-theta_est * 0.1) * X[:-1]- 0.1 *theta_est * mu_est) 
    t = np.linspace(-5, 5, 1000)

    fig, ax = plt.subplots()
    ax.hist(emp_dist, density = True, bins = 500, label = 'Empirical Distribution')
    ax.plot(t, sp.stats.norm.pdf(t), label = r'$Z \sim N(0, 1)$')

    print(sp.stats.norm.fit(emp_dist), sp.stats.skew(emp_dist))
    ax.set(xlim = [-5, 5], ylim = [-0, 0.5], title = 'Empirical dist. of ' + r"$\frac{1}{\sqrt{dt \sigma^2}}(X_t - (1-\theta dt)X_{t-1})$")
    ax.legend()

def pre_post_gof(id, log = True, bs_num = 0, err = False, v = False, r_corrected = True):
    data = get_rolling(id, log = log)['diff'].dropna()
    start = pos_drop_zero(to_df(pos[id])).index[0]

    pre_train = data[:start]
    post_train = data[start:]
    N = len(post_train)
    dt = 0.1
    stepsize = 0.01

    cdf_ran, cdf_pre, theta_pre, sigma_pre = gen_cdf(pre_train.values, bs_num=bs_num, stepsize=stepsize, v = v)
    dummy_ran, cdf_post, theta_post, sigma_post = gen_cdf(post_train.values, bs_num=bs_num, stepsize=stepsize)

    pdf_pre = 1/stepsize * (cdf_pre[1:]-cdf_pre[:-1])
    pdf_post = 1/stepsize * (cdf_post[1:]-cdf_post[:-1])

    if r_corrected == True:
        r_sm = vol_metrics['r_sigma pred.'][id]
        print(r_sm)
        pdf_pre = pdf_pre / r_sm
        cdf_ran = cdf_ran * r_sm
        print(np.sum(pdf_pre) * (stepsize * r_sm))
    
    fig, (ax1, ax2) = plt.subplots(nrows = 2)
    ax1.plot(cdf_ran[1:], pdf_pre, label = 'pre')
    ax1.plot(dummy_ran[1:], pdf_post, label = 'post')

    ax2.plot(cdf_ran[1:], pdf_pre, label = 'pre')
    ax2.plot(dummy_ran[1:], pdf_post, label = 'post')
    ax1.set(xlim = [-4, 4], title = 'smoothed emp. dist. for case {0}, pre buy and post buy'.format(id*10))
    ax2.set(xlim = [-0.2, 0.2])
    ax2.axhline(1, c='black')
    ax1.legend()
    if r_corrected==False:
        diff = pdf_pre/pdf_post-1
        mae = 100 * np.sum(np.abs(pdf_pre - pdf_post))/np.sum(pdf_post)
        ax1.plot(cdf_ran[1:], diff, label = 'prop. diff')
        ax2.plot(cdf_ran[1:], diff, label = 'prop. diff')

        if err == True:
            return mae
        if v == True:
            print('mae (% area): ', mae)