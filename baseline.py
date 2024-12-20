from util_funcs import *

#   wrappers on top of original pnl_fast for convenience; does this slow things down after precompilation? 
#   it does but negligibly 
vol = 5000

def pnl_v(i): 
    print(i, getname(pos[i]))
    pos_df = pos_drop_zero(to_df(pos[i]))
    td_df = to_df(td[i])
    return np.array(pnl_fast(pos_df, td_df, vol))
"""
#   generates from all of the csvs then saves the numpy array
res = np.zeros((len(pos), 5))
for i in range(len(pos)):
    res[i, :] = pnl_v(i)

np.save("all_pnl_fixed_vol_1000.npy", res)
"""

#   need to add volume column by doing avg bid * bought + avg ask * sold and rounding to int

baseline_res = np.load("files/all_pnl_fixed_vol_1000.npy")
df_res = pd.DataFrame(baseline_res, columns = ['sold', 'bought', 'net', 'avg bid', 'avg ask'], index = [getname(pos[i]) for i in range(len(pos))])
df_res['avg bid'] = df_res['avg bid'] * -1 
df_res = df_res[['bought', 'sold', 'net', 'avg bid', 'avg ask']]

vol = baseline_res[:, 0] / baseline_res[:, 4]
df_res.insert(5, 'vol traded', vol.astype(int))

#   calculates some numbers
winrate = len(np.flatnonzero(df_res['net'] > 0))/len(pos)
loserate = len(np.flatnonzero(df_res['net'] <= 0))/len(pos)

total_profit = np.sum(df_res['net'])
avg_earnings = np.mean(df_res['net'])
earnings_given_w = np.mean(baseline_res[np.flatnonzero(df_res['net'] > 0), 2])
loss_given_l =  np.mean(baseline_res[np.flatnonzero(df_res['net'] < 0), 2])
pnl_ratio = -1 * earnings_given_w/loss_given_l
#1.745810678636424
