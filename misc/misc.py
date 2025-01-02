from utils.util_funcs import * 

def bid_ask_spread(td_fname):
    td = to_df(td_fname)
    bids = td['bid_p1'].values
    asks = td['ask_p1'].values

    bid_vols = td['bid_v1'].values
    ask_vols = td['ask_v1'].values
    spread = asks - bids

    return np.min(spread), np.max(spread), np.mean(spread), np.mean(bid_vols), np.mean(ask_vols)

#   checks volume dependence on net revenue (should be ~ linear since price diff on limit orders should be relatively negligible)
def test_vol(pos, td):
    vol_vals = np.arange(1000, 10000, 500)
    profits = [pnl_fast(pos, td, volume)[2] for volume in vol_vals]

    fig, ax = plt.subplots()
    ax.plot(vol_vals, profits)