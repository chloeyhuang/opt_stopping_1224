import iisignature as iis

from utils.analysis import * 
from utils.OU import gen_OU_sample
from utils.pytorch_opt import adam_opt

def gen_partial_sigs(trade_data, pos_data, time, N, M, depth, interval = 1, emp = True, v = False, b_cor = 1):
    paths, time_index = gen_OU_sample(trade_data, pos_data, time, N = N, M = M, emp = emp, interval = interval, log = True)

    #estimated the 1/n(Y_i+n - Y_i) term by b_cor * 1.8 * (X_i+n - E(X))
    payoffs = paths[:, 1:] - paths[:, :-1] + b_cor * 1.8/(500/interval) * (paths[:, :-1] - np.mean(paths[:, :-1], axis=1, keepdims = True))

    sl = iis.siglength(2, depth)

    partial_sigs = np.zeros((M, N-1, sl+1))
    partial_sigs[:, :, 0] = 1
    
    ticker = tqdm(range(M)) if v == True else range(M)

    for i in ticker:
        aug_path = np.vstack((time_index, paths[i, :])).transpose()
        partial_sigs[i, :, 1:] = iis.sig(aug_path, depth, 2)

    return partial_sigs, payoffs

#   monte carlo it to remedy sensitivity to initial l choice
#   final params decided were M = 100, depth = 2, k = 1, int = 10/20, (N = 600/900)
def get_opt_stopping_time_batched(trade_data, pos_data, time, N, M, depth, k = 0.4, interval = 1, emp=False, v= False, batches = [15, 2]):
    ticker = tqdm(range(batches[0])) if v==True else range(batches[0])
    res = [gen_partial_sigs(trade_data, pos_data, time, N, M, depth, interval, emp) for i in ticker]

    def batch(i):
        ps, pf = res[i]
        losses = np.zeros(batches[1])
        stop_times = np.zeros(batches[1])

        for i in range(batches[1]):
            l_opt, loss = adam_opt(ps, pf, k, epochs = 700)

            #   performs inner product and squares it 
            in_prod = ps @ l_opt
            sig_dist = in_prod**2

            sig_dist_cumsum = np.apply_along_axis(np.cumsum, -1, sig_dist)
            opt_stopping_times = np.argmax(sig_dist_cumsum > k, axis = 1) + 1

            stop_times[i] = np.mean(opt_stopping_times)
            losses[i] = loss
        return np.array([losses, stop_times]).transpose()

    res = [batch(i) for i in ticker]
    all_res = np.vstack(tuple(res))
    stop_times = all_res[:, 1]
    losses = all_res[:, 0]
    
    st_nz = np.flatnonzero(stop_times != np.min(stop_times))
    losses_nz = losses[st_nz]
    st_nz = stop_times[st_nz]

    losses_min = np.argsort(losses_nz)[:max(batches[1], 2)]

    if v == True:
        print(stop_times.reshape(batches)) 
        print(losses.reshape(batches))
    return st_nz[losses_min]