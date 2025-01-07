import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
    
#   defines pytorch model
class Model(nn.Module):
    def __init__(self, l, ps, pf, k):
        super(Model, self).__init__()
        self.l = nn.Parameter(torch.clone(torch.tensor(l)))
        self.l.requires_grad = True
        self.ps = torch.from_numpy(ps)
        self.pf = torch.from_numpy(pf)
        self.k = k

    def forward(self):
         #   inner product between l and signatures  M x (N-1)
        in_prod = self.ps @ self.l 
        #   inner prod squared M x (N-1)
        sig_dist = in_prod**2
        #   partial sums of square of inner prod for increasing time; M x (N-1)
        partial_sig_dist_sum = torch.cumsum(sig_dist, dim = -1)
        #   G_k(x) = 1 - 1/(1+np.exp(-20(x-k)))
        G = lambda x: 1/2 * (1 - torch.tanh(40 * x))
        partial_cdf_sums = G(partial_sig_dist_sum)
        losses = self.pf @ partial_cdf_sums.t()

        return -(losses.diag()).mean()

#   adam optimisation
def adam_opt(ps, pf, k, epochs = 1000, lr = 0.01, sc = 0.005, v = False):
    l0 = np.random.normal(size = ps.shape[2], scale = sc)
    model = Model(l0, ps, pf, k)
    if v == True:
        print('initial loss:', float(model.forward()))

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.2, 0.2), weight_decay=0.3)
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

    return param_hist[np.argmin(loss_hist), :], float(np.min(loss_hist))
    #return param_hist[np.argmin(loss_hist), :], float(np.min(loss_hist))
