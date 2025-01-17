from ost import * 

def sell_times(volume, ref_time, window, freq):
    ref_time = pd.Timestamp(ref_time)
    sell_times = pd.date_range(ref_time - window * pd.Timedelta(freq), ref_time + window * pd.Timedelta(freq), freq = freq, inclusive = 'right')

    length = len(sell_times)
    vol = volume/length
    return pd.Series(data = vol *np.ones(length), index = sell_times)

#   archived; dont use this
def test_buy(id, plot = False, simple = False):
    trade_data = to_df(td[id])
    p = trade_data['ask_v1'].iloc[5000::5]
    pos_times = to_df(pos[id])
    scale = (10**4 * np.std(get_returns(trade_data)[:p.index[0]]))**2
    print(scale)

    score = fast_mmtm(trade_data, p, weight = 2000, vol_scale = True)

    pos_range = pd.date_range(pos_times.index[0], pos_times.index[-1], freq = '100ms')

    entry_time = 0
    exit_time = 0

    entry_lim = 20
    exit_lim = 10

    start = -2.5
    stop = -1
    strong_start = -15

    entry_timer = 0
    exit_timer = 0

    for t in score.index:
        if score[t] < start and t in pos_range:
            if entry_timer < entry_lim:
                entry_timer += 1
            elif entry_timer == entry_lim and entry_time == 0:
                entry_time = t
            else:
                pass
        elif score[t] < strong_start:
            if entry_time == 0:
                entry_time = t
                entry_timer = entry_lim
            else:
                pass
        
        if score[t] > stop and entry_time != 0:
            if exit_timer == exit_lim: 
                exit_time = t
                break
            elif exit_timer < exit_lim:
                exit_timer += 1
    
    if exit_time == 0:
        exit_time = score.index[-1]
    
    if simple == True:
        if entry_time == 0:
            return -1
        else:
            return len(score.loc[entry_time:exit_time:5])

    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(trade_data['bid_p1'])
        ax1 = ax.twinx()
        ax1.axhline(stop , c= 'black', label = 'exit')
        ax1.axhline(start, c= 'grey', label = 'entry')
        ax1.axhline(strong_start, c = 'g', label = 'strong entry')
        ax1.plot(score , c = 'g')
        if entry_time != 0:
            ax1.plot(score.loc[entry_time:exit_time] , c = 'r')
        else:
            print('no times')
        ax1.set(xmargin=0)
        ax.set(xmargin =0)
        ax1.legend()
    if entry_time == 0:
        print('no times')
        return score
    else:
        return score.loc[entry_time:exit_time:5]

#   use this
def buy_times_by_case(id):
    entry_thres =  -2.5
    strong_entry_thres = -10
    exit_thres = -1

    entry_timer = 0
    exit_timer = 0

    trade_data = to_df(td[id])
    pos_data = to_df(pos[id])
    start = trade_data.index[0]
    end = trade_data.index[-1]

    buy_start_time = 0
    exit_time = 0

    entry_lim = 20
    exit_lim = 10

    score_times = trade_data.loc[start + pd.Timedelta(minutes=12):end - pd.Timedelta(minutes=5)]

    scores = fast_mmtm(trade_data, score_times, weight = 2000, vol_scale=True)

    if np.min(scores) > strong_entry_thres and len(pos_data.index) < entry_lim:
        return pd.Series(data = ['no times'], index = [scores.index[0]])

    trades = pd.Series()

    for time in scores.index:
        score = scores[time]
        pos_signal = True if time in pos_data.index else False

        if score < entry_thres and pos_signal == True:
            if entry_timer < entry_lim:
                entry_timer += 1
                trades.loc[time] = 0
            elif entry_timer == entry_lim and buy_start_time == 0:
                buy_start_time = time
                trades.loc[time] = abs(score)
            
        elif score < strong_entry_thres and buy_start_time == 0:
            buy_start_time = time
            entry_timer = entry_lim
            trades.loc[time] = abs(score)

        if score > exit_thres and entry_timer != 0:
            if exit_timer == exit_lim:
                exit_time = time
                trades.loc[time] = -1
                break

            elif exit_timer < exit_lim:
                exit_timer += 1
                trades.loc[time] = 0
        
        if score < exit_thres and entry_timer == entry_lim:
            trades.loc[time] = abs(score)
        
    return trades

