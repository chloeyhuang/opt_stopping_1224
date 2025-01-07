from ost import * 

def test(id, k, interval, sc = 1):
    t = to_df(td[id])
    p = to_df(pos[id])
    tm = p.index[0]
    return get_opt_stopping_time_batched(t, p, tm, 300, 50, 2, k, interval, v = True, sc = sc)