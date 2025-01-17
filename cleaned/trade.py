from optimal_stopping_times import * 

class trade:
    def __init__(self, time, params):
        self.open = False
        self.close = False
        self.time = time
        self.freq = params['freq']

        self.rolling_window = params['rolling_window']

        self.entry_timer = 0
        self.exit_timer = 0

        self.entry_lim = params['entry_lim'] #  20
        self.exit_lim = params['exit_lim'] #    10

        self.vol_multiplier = params['vol_multiplier'] #    like 1k?
        self.max_order_size = params['max_order_size'] #    like 7k?

        self.buy_start_time = 0
        self.buy_stop_time = 0

        self.entry_thres = params['entry_thres'] #  -2.5
        self.strong_entry_thres = params['strong_entry_thres'] # -10
        self.exit_thres = params['exit_thres'] # -1

        self.sell_times = pd.Series(None)
        self.log = pd.DataFrame(data = None, columns = ['action', 'price', 'vol'])

    def get_data(self):
        #   returns past rolling_window of close data
        return pd.DataFrame(None)
    
    def score(self, data): #    data is past 18 minutes of data; required is close data and trade volume
        data = data.loc[self.time-pd.Timedelta(minutes=18):self.time]
        weight = 2000
        
        norm_const = weight/10 *(1-np.exp(-1000/weight))
        vol_scaled = (data['trade_volume']/data['trade_volume'][0]).apply(lambda t: max(2, t)).iloc[:-1]
        close = data['close']
        times = data.index
        diff = close[self.time] - close
        tdiff = (times - self.time).to_numpy(dtype = 'float64')

        gradient = (-diff/tdiff).dropna()

        return 10000 * np.sum(gradient * vol * np.exp(-10 * np.abs(np.trim_zeros(tdiff)/weight)))/norm_const
    
    def buy(self, data, pos_signal):
        #   explanation of entry logic:
        #   - if below entry thres for first time, start timer
        #   - once below entry thres for entry_lim times, start buying 
        #   - if below strong entry thres, start buying immediately 
        #   - if above exit thres, start timer 
        #   - once above exit thres for exit_lim times, stop buying and close 

        score = self.score(data)
        if score < self.entry_thres and pos_signal == True:
            if self.entry_timer < self.entry_lim:
                self.entry_timer += 1
                return [0, 0]
            elif self.entry_timer == self.entry_lim and self.buy_start_time == 0:
                self.buy_start_time = self.time
                self.open = True
                return [1, abs(score)]
            
        elif score < self.strong_entry_thres and self.buy_start_time == 0:
            self.buy_start_time = self.time
            self.entry_timer = self.entry_lim
            self.open = True
            return [1, abs(score)]

        if score > self.exit_thres and self.open == True:
            if self.exit_timer == self.exit_lim:
                self.exit_time = self.time
                self.open = False
                self.close = True
                return [0, -1]
            elif self.exit_timer < self.exit_lim:
                self.exit_timer += 1
                return [0, 0]
        
        if self.open == True and score < self.exit_thres:
            return [1, abs(score)]
    
    def alt_signal(self, data):
        #   original signal stuff here
        return True
    
    def get_sell_times(self, volume, ref_time):
        ref_time = pd.Timestamp(ref_time)
        window = min(50, int(volume/self.max_order_size))
        sell_times = pd.date_range(ref_time - window * pd.Timedelta(self.freq), ref_time + window * pd.Timedelta(self.freq), freq = self.freq, inclusive = 'right')

        length = len(sell_times)
        vol = volume/length
        return pd.Series(data = vol *np.ones(length), index = sell_times)    

    def sell(self):
        if self.time in self.sell_times.index:
            return [-1, sell_times.loc[self.time]]
    
    def opt_stopping(self, data):
        return get_opt_stopping_time_batched(data, self.time, 600, 50, 2, 1, 20, batches = [12, 3])
    
    def update_time(self):
        return None
        #   update the time to be correct; how do i do this? 
    
    def get_price(self):
        #   get the current price?
        return None

    def vol(self, score):
        return min(self.vol_multiplier * score, self.max_order_size)
    
    def next(self, data, time_correction):
        #   buy according to scoring
        if self.close == False:
            alt_sig = alt_signal(self, data)
            action = buy(self, data, altsig)
            if action[0] == 1:
                action[1] = self.vol(action[1])
                self.log[self.time] = [1, self.get_price(), action[1]]
        
        #   calculate opt stopping and get sell times
        if action == [0, -1]:
            stopping_times = self.opt_stopping(data)

            if np.max(stopping_time) < 20:
                warnings.warn('{0}: excessively short stopping time of {1}'.format(self.time, stopping_time))
            stopping_times = np.round(stopping_times, decimals=1)

            ref_times = self.time + stopping_times * pd.Timedelta(seconds=2)
            volume = np.sum(self.log['volume'])/len(stopping_times)
            self.sell_times = pd.concat([self.get_sell_times(volume, ref_time, 10) for ref_time in ref_times])

        #   sell according to sell times
        if sell.close == True:
            action = sell()
            self.log[self.time] = [-1, self.get_price, action[1]]
        
        self.update_time()
        return action