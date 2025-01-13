from optimal_stopping_times import *

#   template for getting optimal buy volume and time for fixed time; this doesn't actually run because fast_mmtm still depends on pos data for the moment
def buy(trade_data, time, thres):
    unscaled_score = fast_mmtm(trade_data, time, weight = 2000)
    sc = get_returns(trade_data)[:time] #    returns for volatility
    score = unscaled_score/ (10**4 * np.std(sc))**2

    if score < thres:
        opt_stopping = get_opt_stopping_time_batched(trade_data,time, 600, 50, 2, 1, interval=20, batches=[15, 3])
        #   returns the optimal stopping time in seconds (as interval = 20 * 100ms) and the abs value of score as the multiplier for volume
        return opt_stopping * 2, np.abs(score)
    else: 
        return -1, 0