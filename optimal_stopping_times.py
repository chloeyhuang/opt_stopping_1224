import iisignature as iis

from analysis import * 
from OU import gen_OU_sample
from pytorch_opt import adam_opt

def aug_sig(id, end_time = 'pos', log = True, depth = 3, price = False, structure = 1, interval = 1):
    data = get_rolling(id, log = log, interval = interval).dropna()
    if end_time == 'pos':
        end = to_df(pos[id]).index[0]
        data = data[:end]
    else:
        data = data[:end_time]
    
    n = len(data.index)

    if price == False:
        data = data['diff'].values.reshape(n, 1)
    else:
        data = data[['price', 'diff']].values.reshape(n, 2)

    index = np.arange(0, n).reshape(n, 1) * dt
    aug_path = np.hstack((index, data))
    signature =  iis.sig(aug_path, depth, structure)
    return signature

def gen_partial_sigs(id, time, N, M, depth, interval = 1, emp = True, v = False, b_cor = 1):
    paths, time_index = gen_OU_sample(id, time, N = N, M = M, emp = emp, v= v, interval = interval)

    #estimated the 1/n(Y_i+n - Y_i) term by b_cor * 1.5 * (X_i+n - E(X))
    payoffs = paths[:, 1:] - paths[:, :-1] + b_cor * 1.5/500 * (paths[:, :-1] - np.mean(paths[:, :-1], axis=1, keepdims = True))

    sl = iis.siglength(2, depth)

    partial_sigs = np.zeros((M, N-1, sl+1))
    partial_sigs[:, :, 0] = 1
    
    ticker = tqdm(range(M)) if v == True else range(M)

    for i in ticker:
        aug_path = np.vstack((time_index, paths[i, :])).transpose()
        partial_sigs[i, :, 1:] = iis.sig(aug_path, depth, 2)

    return partial_sigs, payoffs

def gen_partial_sigs_real_data(id, N, depth, interval = 1):
    pos_data = to_df(pos[id])
    start = pos_data.index[0]
    data = get_rolling(id, interval = interval, log = True).dropna()[:start]
    payoffs = ((data['price'] - data['price'][0]).values)[-N:]
    path = (data['diff'].values)[-N:]
    sl = iis.siglength(2, depth)

    time_index = np.arange(0, N * (interval * 0.1), interval * 0.1)
    partial_sigs = np.zeros((1, N-1, sl+1))
    partial_sigs[0, :, 0] = 1
    partial_sigs[0, :, 1:] = iis.sig(np.vstack((time_index, path)).transpose(), depth, 2)

    return partial_sigs, payoffs[:-1]

#   calculates optimal stopping time (single sample, no mc validation)
def get_opt_stopping_time(id, N, M, depth, k, interval = 1, emp=True, v= False):
    partial_sigs, payoffs = gen_partial_sigs(id, N, M, depth, emp = emp, v = v, interval = interval)

    l_opt, loss = adam_opt(partial_sigs, payoffs, k, epochs = 500, v = v)
    print(loss)
    #print(l_opt)

    #   performs inner product and squares it 
    in_prod = np.apply_along_axis(lambda x: np.inner(l_opt, x), -1, partial_sigs)
    sig_dist = in_prod**2

    sig_dist_cumsum = np.apply_along_axis(np.cumsum, -1, sig_dist)
    opt_stopping_times = np.argmax(sig_dist_cumsum > k, axis = 1) + 1

    #opt_stopping_time = dt * np.argmin(mean_sig_dist_cumsum < k)
    return np.mean(opt_stopping_times)

#   using AdamW: N = 600, interval = 10(1s), k = 5, depth 2 
def get_opt_stopping_time_from_real(id, N, depth, k, interval = 1, v= False):
    partial_sigs, payoffs = gen_partial_sigs_real_data(id, N, depth, interval = interval)
    losses = []
    stop_times = []
    for i in tqdm(range(20)):
        l_opt, loss = adam_opt(partial_sigs, payoffs, k, epochs = 3000, v = v)
        losses.append(loss)
        l_opts.append(l_opt)
        #print(loss)
        #print(l_opt)
        #print('-')

        #   performs inner product and squares it 
        in_prod = np.apply_along_axis(lambda x: np.inner(l_opt, x), -1, partial_sigs)
        sig_dist = in_prod**2

        sig_dist_cumsum = np.apply_along_axis(np.cumsum, -1, sig_dist)
        opt_stopping_times = np.argmax(sig_dist_cumsum > k, axis = 1) + 1
        stop_time = np.mean(opt_stopping_times)
        stop_times.append(stop_time)
        #opt_stopping_time = dt * np.argmin(mean_sig_dist_cumsum < k)

    print(np.mean(sig_dist_cumsum, axis = 0)[:10])
    losses = np.array(losses)
    stop_times = np.array(stop_times)
    print(stop_times[np.flatnonzero(stop_times != 1)[np.argmin(losses[np.flatnonzero(stop_times != 1)])]])
    return stop_times

#   monte carlo it to remedy sensitivity to initial l choice
#   final params decided were M = 50, depth = 2, k = 10, int = 10, (N = 600/900)

def get_opt_stopping_time_batched(id, time, N, M, depth, k, interval = 1, emp=True, v= False, batches = [10, 2]):
    ticker = tqdm(range(batches[0])) if v==True else range(batches[0])
    res = [gen_partial_sigs(id, time, N, M, depth, interval, emp) for i in ticker]

    def batch(i):
        ps, pf = res[i]
        losses = np.zeros(batches[1])
        stop_times = np.zeros(batches[1])

        for i in range(batches[1]):
            l_opt, loss = adam_opt(ps, pf, k, epochs = 1500, v = v)

            #   performs inner product and squares it 
            in_prod = ps @ l_opt
            sig_dist = in_prod**2

            sig_dist_cumsum = np.apply_along_axis(np.cumsum, -1, sig_dist)
            opt_stopping_times = np.argmax(sig_dist_cumsum > k, axis = 1) + 1

            stop_times[i] = np.mean(opt_stopping_times)
            losses[i] = loss
        return np.array([losses, stop_times]).transpose()

    all_res = np.vstack(tuple([batch(i) for i in ticker]))
    stop_times = all_res[:, 1]
    losses = all_res[:, 0]
    
    st_nz = np.flatnonzero(stop_times != np.min(stop_times))
    losses_nz = losses[st_nz]
    st_nz = stop_times[st_nz]

    losses_min = np.argsort(losses_nz)[:max(batches[1], 2)]
    if v == True:
        print(stop_times.reshape(batches)) 
        print(losses.reshape(batches))
    return np.mean(st_nz[losses_min])