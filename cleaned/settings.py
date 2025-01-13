PATH = '/home/leewind/mnt/opt_stopping_1224/'

import os
os.chdir(PATH)

train_header = '/mnt/data/'
header = '/home/leewind/mnt/opt_stopping_1224/'

train_suffix = 'bs_data_241016/'
train_4h_suffix = 'bs_data_4h_241113/'
#   time diff in seconds (baseline is 10 minutes) - need to change this so that I dont rely on a global variable and instead can do variable sell time
tdiff = 600
latex = False