from multiprocess import Pool
import numpy as np
import torch.multiprocessing as mp

from optimal_stopping_times import *
from pytorch_opt import *

def test(i):
    #print(i)
    return i**2

ps, pf = gen_partial_sigs(10, 600, 10, 2, 50, 10)

def test2(i):
    print(i)
    return adam_opt(ps, pf, 0.1, epochs = 2000, v = True)

p = Pool(5)
#res = p.map(test, range(5))

res = [test(i) for i in range(5000)]