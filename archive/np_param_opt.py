
from optimal_stopping_times import * 

def loss_by_sim_paths(id, l, M, depth, k):
    partial_sigs, payoffs = gen_partial_sigs(id, M, depth)
    if len(l) != partial_sigs.shape[2]:
        return "err: l wrong length, expected length {0}".format(partial_sigs.shape[2])
    
    in_prod = np.apply_along_axis(lambda x: np.inner(l, x), -1, partial_sigs)
    
    sig_dist = in_prod**2
    partial_sig_dist_sum = np.apply_along_axis(np.cumsum, -1, sig_dist)
    #   G_k(x) = 1 - 1/(1+np.exp(-20(x-k)))
    partial_cdf_sums = 1-1/(1+np.exp(-20*(partial_sig_dist_sum-k)))
    losses = np.einsum('ij, ij -> i', payoffs, partial_cdf_sums)

    return -np.mean(losses)

def loss_minimisation(partial_sigs, payoffs, M, depth, k, layers = 120):
    def loss(l):
        #   inner product between l and signatures 
        in_prod = np.apply_along_axis(lambda x: np.inner(l, x), -1, partial_sigs)
        #   inner prod squared
        sig_dist = in_prod**2
        #   partial sums of square of inner prod for increasing time
        partial_sig_dist_sum = np.apply_along_axis(np.cumsum, -1, sig_dist)
        #   G_k(x) = 1 - 1/(1+np.exp(-20(x-k)))
        partial_cdf_sums = np.apply_along_axis(lambda x: 1-1/(1+np.exp(-20*(x-k))), -1, partial_sig_dist_sum)
        losses = np.einsum('ij, ij -> i', payoffs, partial_cdf_sums)

        return -np.mean(losses)

    def grad(l):
        #   inner product between l and signatures 
        in_prod = np.apply_along_axis(lambda x: np.inner(l, x), -1, partial_sigs)
        #   inner prod squared
        sig_dist = in_prod**2
        #   partial sums of square of inner prod for increasing time M x (N-1)
        partial_sig_dist_sum = np.apply_along_axis(np.cumsum, -1, sig_dist)

        G_dash = lambda x: 20 * (1 + np.exp(-20*(x-k)))**(-2)
        #   partial signatures squared, M x (N-1) x (siglen+1)
        partial_sig_sq = partial_sigs**2

        #   cum sum of partial signatures in l_k, sum_i=0^j (X_i)k^2, M x (N-1) x (siglen+1)
        ps_sig_dist_cum =  np.apply_along_axis(np.cumsum, 1, partial_sig_sq)

        #   G'(sum over i (inner(l, x_i))), M x (N-1)
        partial_pdf_sums = np.apply_along_axis(lambda x: G_dash(x), -1, partial_sig_dist_sum)

        #   (Y_j+1 - Y_j) * G'(sum), M x (N-1)
        partial_payoff_scaled = payoffs * partial_pdf_sums

        #   putting it together, M x (siglen + 1)
        grad_summed = np.einsum('ijk, ij -> ik',  2*ps_sig_dist_cum, partial_payoff_scaled)

        return np.mean(grad_summed, axis=0) * l
    
    def adam_opt(lr_max = 0.05, betas = (0, 0.2), eps = 1e-8, restart = 80, layers = layers):
        learn_rate = lambda T: 0.00005 + 1/2 * (lr_max - 0.00005) * (1 + np.cos((T % restart)/restart * np.pi))

        n = partial_sigs.shape[2]
        m = np.zeros(n)
        v = np.zeros(n)

        b1 = betas[0]
        b2 = betas[1]

        losses = np.zeros(layers)
        params = np.zeros((layers, n))

        l = np.random.normal(size=n)
        print(loss(l))

        def adm_update(m_prev, v_prev, l, t):
            lr = learn_rate(t)
            g = grad(l)
            m = b1 * m_prev + (1-b1)*g
            v = b2 * v_prev + (1-b2) * g**2

            m_hat = m/(1-b1**(t+1))
            v_hat = v/(1-b2**(t+1))

            l_opt = l - lr * m_hat / (np.sqrt(v_hat) + eps)
            return l_opt, m, v
        
        for t in tqdm(range(layers)):
            l, m, v = adm_update(m, v, l, t)
            cur_loss = loss(l)
            if abs(cur_loss) <= 1e-8:
                m = 0
                v = 0
                print('soft restart')
                l = l + np.random.normal(scale = 0.5, size = len(l))
            
            losses[t] = cur_loss
            params[t, :] = l

        l_min = np.argmin(losses)
        #plt.plot(np.arange(len(losses)), losses)
        return params[l_min, :], losses[l_min]
    
    return adam_opt(layers=layers)
