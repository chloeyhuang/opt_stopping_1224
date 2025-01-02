from utils.analysis import *
import optuna

#   stuff for vol modelling
scaling = lambda x: np.abs(np.tanh((x-1)))
vol_metrics = pd.read_csv('files/vol_metrics.csv')

#   analysis / diagnostics
def plot_score_vol_cor():
    res = sp.stats.linregress(all_scores['score'], all_scores['mean volume'])
    x = np.linspace(-400,100)
    fig, ax = plt.subplots()
    ax.scatter(all_scores['score'], all_scores['mean volume'], s= 2, label = 'score vs mean volume')
    ax.plot(x, res.intercept - 30000 + res.slope*x, 'g', label = '(intercept adjusted) linear regression on score vs mean volume')
    ax.set(xlim = [-400, 100], ylim = [0, 200000], xlabel = 'score', ylabel = 'mean volume')
    ax.legend(loc = 'lower left')

def vol_ratio(id):
    data = get_returns(td[id])
    pos_data = to_df(pos[id]).index
    start = pos_data[0]
    end = pos_data[-1]

    vol_before = np.std(data[:start][-6000:])
    vol_after = np.std(data[end:][:6000])

    return vol_after/vol_before

def est_vol_from_post_data(id, n):
    data = get_returns(td[id])
    pos_ind = to_df(pos[id]).index
    start = pos_ind[0]
    end = pos_ind[-1]

    vol_before= np.std(data[:start])
    vol_after = np.std(data[end:])
    vol_est = np.std(data[end:][:n])

    return vol_est/vol_before

def pre_vol(id, start = True):
    data = get_returns(td[id])
    start = to_df(pos[id]).index[0] if start == True else to_df(pos[id]).index[-1] 
    return np.std(data[:start][-6000:])

def post_vol(id):
    data = get_returns(td[id])
    pos_d = to_df(pos[id]).index
    return np.std(data[pos_d[-1]:][:6000])

#vr = pd.Series(data = [vol_ratio(id) for id in tqdm(range(523))])
vr = np.load('files/vr.npy')
pv = np.load('files/pv.npy')
ev = np.load('files/ev.npy')
shortm = np.load('files/shortmom.npy')

#   optuna hyperparameter training for correlation
def objective(trial):
    x = 1/(1000 * pv)
    y = -shortm/10

    a = trial.suggest_float('a',-1,1)
    p = trial.suggest_float('p', -1, 2)
    c = trial.suggest_float('c', -1, 1)

    res = (np.abs(a*x + (1-a)*y))**p + c*x

    return -abs(np.corrcoef(res, vr)[1, 0])
"""
study = optuna.create_study()
study.optimize(objective, n_trials=400)
"""

def est_vol_ratio(pre_vol, short_mom):
    x = 1/(1000 * pre_vol)
    y = -short_mom/10
    a = 0.128
    p = 1.174
    c = -0.106
    eq = (np.abs(a * x + (1-a) * y)**(p) + c * x)/100

    return 0.7604 + 20.534 * eq
