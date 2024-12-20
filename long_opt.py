from optimal_stopping_times import *
buy_p = np.zeros(523)
sell_p = np.zeros(523)
for i in range(523):
    td_c = to_df(td[i])
    pos_c = to_df(pos[i])
    print(i)
    
    start = pos_c.index[int((len(pos_c.index)/2))]
    stop = start + timedelta(seconds = get_opt_stopping_time(i, 10, 2, 15) * 10)
    print(stop)
    buy_p[i] = td_c.at_time(start)['ask_p1']
    sell_p[int(i)] = td_c.at_time(stop)['bid_p1']
