from utils.optimal_stopping_times import * 
from utils.pytorch_opt import * 
import optuna

#   hyperparameter tuning
#ps, pf = gen_partial_sigs(100, 20, 3)
adm = lambda lr, sc: adam_opt(ps, pf, 20, 3, 30, lr = lr, sc = sc)

#   adam optimisation
def adam_opt_all(ps, pf, k, epochs = 2000, lr = 0.007029, sc = 0.035949, b1 = 0.2, b2 = 0.2, weight_decay = 0.08, v = False):
    l0 = np.random.normal(size = ps.shape[2], scale = sc * k)
    model = Model(l0, ps, pf, k)
    if v == True:
        print('initial loss:', float(model.forward()))

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = epochs)

    loss_hist = np.zeros(epochs)
    param_hist = np.zeros((epochs, len(l0)))
    
    ticker = range(epochs) if v==False else tqdm(range(epochs))

    for epoch in ticker:
        optimiser.zero_grad() # don't use old gradients
        out = model()
        loss = out

        loss_hist[epoch] = float(loss)
        param_hist[epoch, :] = np.array(model.l.data)

        loss.backward() # compute gradients wrt loss
        optimiser.step() # update parameters    
        scheduler.step() # update optimiser parameters

    if v == True:
        fig, ax = plt.subplots()
        ax.plot(np.arange(epochs), loss_hist)
        ax.set(xmargin = 0)

    return param_hist[-1, :], float((loss_hist[-1]))
    #return param_hist[np.argmin(loss_hist), :], float(np.min(loss_hist))


def objective_single(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.5)
    sc = trial.suggest_float('sc', 10**(-4), 0.1)

    return adm(lr, sc)[1]

#study = optuna.create_study()
#study.optimize(objective_single, n_trials=50)

#print(study.best_params)
#for payoff = price - price.rolling.mean: {'lr': 0.011806724311329723, 'sc': 0.009476886666744807}
#for adjusted payoff (17/12/24) {'lr': 0.4726343229334647, 'sc': 0.09216927807211477}

def objective_real_data(trial):
    b1 = trial.suggest_float('b1', 0.0, 0.6)
    b2 = trial.suggest_float('b2', 0.0, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.5)
    rand_sample = np.random.randint(0, high=len(pos))
    ps, pf = gen_partial_sigs_real_data(rand_sample, 600, 2, 10)
    mins = np.zeros(20)
    for i in range(20):
        mins[i] = adam_opt_all(ps, pf, 30, b1 = b1, b2 = b2, weight_decay= weight_decay)[1]
    return np.mean(mins) / (np.std(mins) + 10**(-8))
"""
study_r = optuna.create_study()
study_r.optimize(objective_real_data, n_trials=60)

print(study_r.best_params)
"""
def objective_mult(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.3)
    sc = trial.suggest_float('sc', 10**(-4), 0.1)

    mins = np.zeros(20)
    for i in range(20):
        rand_sample = np.random.randint(0, high=len(pos))
        ps, pf = gen_partial_sigs(rand_sample, 20, 3)
        mins[i] = adam_opt(ps, pf, 20, 3, 30, lr = lr, sc = sc)[1]
    
    mn = np.mean(mins)
    std = np.std(mins)
    print(mn, std)
    return (mn) * 10**4
"""
study_m = optuna.create_study()
study_m.optimize(objective_mult, n_trials=12, show_progress_bar=True)

print(study_m.best_params)"""

#18/12/24 {'lr': 0.0070296636274268, 'sc': 0.035949242532471365}
def test(lr = np.random.uniform(low = 0.0001, high = 0.5), sc =np.random.uniform(low = 10**(-4), high = 0.1), n = 20):
    print(lr, sc)

    mins = np.zeros(n)
    for i in tqdm(range(n)):
        ps, pf = gen_partial_sigs(100, 20, 3)
        mins[i] = adam_opt(ps, pf, 20, 3, 30, lr = lr, sc = sc, plot = False)[1]
    
    mn = np.mean(mins)
    std = np.std(mins)
    print(mn, std)
    return (std + mn) * 10**4


#   adam optimisation
def adam_opt_until_convergence(ps, pf, k, lr = 0.007029, sc = 0.035949, v = False):
    M = ps.shape[1]
    depth = ps.shape[2]
    l0 = np.random.normal(size = ps.shape[2], scale = sc * k)
    model = Model(l0, ps, pf, k)
    if v == True:
        print('initial loss:', float(model.forward()))

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.2, 0.1), weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = 10**4)

    loss_hist = np.zeros(10**7)
    param_hist = np.zeros((10**7, len(l0)))
    
    counter = 0
    min_err = 0.00001

    for i in range(3):
        optimiser.zero_grad() # don't use old gradients
        out = model()
        loss = out

        loss_hist[counter] = float(loss)
        param_hist[counter, :] = np.array(model.l.data)

        loss.backward() # compute gradients wrt loss
        optimiser.step() # update parameters   
        scheduler.step() # update optimiser parameters 
        counter += 1 
    
    while abs(loss_hist[counter] - loss_hist[counter-1]) > min_err and counter < 10**6:
        optimiser.zero_grad() # don't use old gradients
        out = model()
        loss = out

        loss_hist[counter] = float(loss)
        param_hist[counter, :] = np.array(model.l.data)

        loss.backward() # compute gradients wrt loss
        optimiser.step() # update parameters    
        scheduler.step() # update optimiser parameters
        counter += 1 
        if counter % 500 == 0:
            print(counter, abs(loss_hist[counter] - loss_hist[counter-1]))

    if v == True:
        fig, ax = plt.subplots()
        ax.plot(np.arange(np.count_nonzero(loss_hist)), loss_hist.nonzero)
        ax.set(xmargin = 0)

    #return param_hist[-1, :], float((loss_hist[-1]))
    return param_hist[np.argmin(loss_hist), :], float(np.min(loss_hist))

