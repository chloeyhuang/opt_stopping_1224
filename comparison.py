from buy_sell_opt import *
"""
buying strategy: 
ret = returns
calculate score = fast_mmtm(id, weight = 2000) / (10**4 * np.std(ret))**2
buy_pos = score <= -3
"""

#   baseline
def buy_thres_baseline(id, thres = -3, p = False, v = False):
    ret = get_returns(td[id])
    score = fast_mmtm(id, weight = 2000) / (10**4 * np.std(ret))**2
    buy_pos = score[score <= thres]
    trade_data = to_df(td[id])

    if len(buy_pos) == 0:
        print('no times in pos that are below threshold. no trades performed for id {0}'.format(id))
        return -np.ones(5)

    sell_pos_ind = buy_pos.index + timedelta(seconds=600)

    buy = np.zeros((len(buy_pos), 2))
    sell = np.zeros((len(buy_pos), 2))

    for i in range(len(buy_pos)):
        buy[i, :] = [fulfill_order(trade_data, buy_pos.index[i], int(100 * abs(buy_pos[i])), -1), int(100 * abs(buy_pos[i]))]
        sell[i, :] = [fulfill_order(trade_data, buy_pos.index[i] + timedelta(seconds=600), int(100 * abs(buy_pos[i])), 1), int(100 * abs(buy_pos[i]))]

    df_buy = pd.DataFrame(data = buy, index = buy_pos.index, columns = ['change', 'vol'])
    df_sell = pd.DataFrame(data = sell, index = sell_pos_ind, columns = ['change', 'vol'])
    if p == True:
        plt.plot(trade_data['bid_p1'])
        plt.plot(df_buy.index, np.mean(trade_data['bid_p1']) * np.ones(len(buy_pos)), marker = '^', c = 'r',alpha = 0.2)
        plt.plot(df_sell.index, np.mean(trade_data['bid_p1']) * np.ones(len(buy_pos)), marker = 'v', c = 'g', alpha = 0.2)
    
    if v == True:
        print(np.sum(sell[:, 0]) + np.sum(buy[:, 0]) )
        return pd.concat((df_buy, df_sell))
    else:
        return np.array([np.sum(buy[:, 0]), np.sum(sell[:, 0]), np.sum(sell[:, 0]) + np.sum(buy[:, 0]), np.sum(buy[:, 1]), 600])

def get_all_res_baseline():
    result = np.zeros((523, 5))
    ids = tqdm(range(np.argmin(result[:, 1] != 0), 523))
    for id in ids:
        result[id, :] = np.array(buy_thres_baseline(id))
    
    return result

res_baseline = np.load(header + 'files/baseline_score_filtered.npy')
df_res_baseline_all = pd.DataFrame(data = np.load(header + 'files/baseline_score_filtered.npy'), columns =  ['bought', 'sold', 'net', 'vol', 'stopping time'])

df_res_baseline = df_res_baseline_all[df_res_baseline_all['vol'] != -1].reset_index()
df_res_baseline['% change'] = -df_res_baseline['net']/df_res_baseline['bought'] * 100

#   calculates some numbers
winrate_baseline = len(np.flatnonzero(df_res_baseline['net'] > 0))/len(df_res_baseline['net'])
loserate_baseline = len(np.flatnonzero(df_res_baseline['net'] <= 0))/len(df_res_baseline['net'])

total_profit_baseline = np.sum(df_res_baseline['net'])
avg_earnings_baseline = np.mean(df_res_baseline['net'])
earnings_given_w_baseline = np.mean(df_res_baseline[df_res_baseline['net'] > 0]['net'])
loss_given_l_baseline =  np.mean(df_res_baseline[df_res_baseline['net'] < 0]['net'])
pnl_ratio_baseline =  -earnings_given_w_baseline/loss_given_l_baseline

general_pnl_baseline = pnl_ratio_baseline * winrate_baseline + loserate_baseline

baseline_unit_change = df_res_baseline['net'] / df_res_baseline['vol']
sharpe_baseline = np.mean(baseline_unit_change)/np.std(baseline_unit_change)

#   opt stopping 
df_res_all = pd.DataFrame(data = np.load(header + 'files/opt_stopping.npy'), columns = ['bought', 'sold', 'net', 'vol', 'stopping time'])

df_res = df_res_all[df_res_all['vol'] != -1].reset_index()
df_res['% change'] = -df_res['net']/df_res['bought'] * 100

#   calculates some numbers
winrate = len(np.flatnonzero(df_res['net'] > 0))/len(df_res['net'])
loserate = len(np.flatnonzero(df_res['net'] <= 0))/len(df_res['net'])

total_profit = np.sum(df_res['net'])
avg_earnings = np.mean(df_res['net'])
earnings_given_w = np.mean(df_res[df_res['net'] > 0]['net'])
loss_given_l =  np.mean(df_res[df_res['net'] < 0]['net'])
pnl_ratio =  -earnings_given_w/loss_given_l
general_pnl = pnl_ratio * winrate + loserate

unit_change = df_res['net']/df_res['vol']

sharpe = np.mean(unit_change)/np.std(unit_change)