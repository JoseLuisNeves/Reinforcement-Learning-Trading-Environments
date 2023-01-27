import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import argrelextrema


class Actions(Enum): 
    Sell = 0
    Buy = 1
    Hold = 2 # Hold will correspond to actions with selected volume equal to 0, so that we increase the probability of the agent to find lucrative actions


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, window_size, frame_bound, initCash, max_transactions, mode = 'train', eps_noise_level=0, trade_fees = 0, wallet_in_obs = False, penalty = 0):

        self.seed()
        self.eps_noise_level = eps_noise_level
        self.penalty = penalty
        self.mode = mode
        if self.mode == 'evaluate':
            self.n_eval_eps = 10 # to get a universal storage for the env even when reseting to enable runs comparison
            self.mean_punish = []
        self.window_size = window_size
        self.trade_fees = trade_fees
        self.initCash = initCash 
        self.wallet_in_obs = wallet_in_obs
        self.max_transactions = max_transactions 
        # we introduce flexible buy/sell volume by creating an action space that includes every possible action for the agent to choose
        # the size of this action space is managed by the parameter max_transactions, that has its ideal size shifting with the relation
        # prices-initcash, because we want to have a minimal size action space without having the limitation of max transactions impacting
        # decisions
        self.frame_bound = frame_bound
        self.num_stocks = 1
        self.signal_features_shape = 2 #[price,diff]
        if self.wallet_in_obs == False:
            # obs = [[price_tminus2,price_diff_tminus3_&_tminus2],[price_tminus1,price_diff_tminus2_&_tminus1],[price_t,price_diff_tminus1_&_t]].flatten()
            self.obs_shape = (window_size*self.signal_features_shape+1,) 
        elif self.wallet_in_obs == True:
            # obs = [[cash,num_stocks_available],[price_tminus2,price_diff_tminus3_&_tminus2],[price_tminus1,price_diff_tminus2_&_tminus1],[price_t,price_diff_tminus1_&_t]].flatten()
            self.obs_shape = ((window_size+1)*self.signal_features_shape+1,)
        # flatten is done by recomendation from the openai gym documentation

        # spaces
        self.action_space = spaces.MultiDiscrete([len(Actions)-1,max_transactions])
        # len(Actions)-1 bescause we are not selecting Hold in the first element but with v=0 in the second
        #self.action_space = spaces.Discrete((len(Actions)-1)*self.max_transactions+1) 
        # the buy and sell actions have to be multiplied by the max transactions allowed, but hold stays just one action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64) 
        # this function builds the action space by getting every possible action for each step

        # episode
        self._start_tick = self.window_size
        self._end_tick = self.frame_bound[1]-self.frame_bound[0]+self.window_size-1
        self._done = None
        self._current_tick = None
        self._action_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = None
        self.wallet = None
        self.volume_history = None
        self.stock_available = None
        self.stock_id = None
        self.num_punishments = None
        self.num_punishments_history = []

    def check_wallet(self):
        return self.wallet

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._action = Actions.Hold.value
        self._action_history = [self._action]*self.window_size
        self._total_reward = 0.
        self._first_rendering = True
        self.history = {}
        self.wallet = []
        self.num_punishments = 0
        self.wallet.append(self.initCash) # cash is the first element of the wallet
        self.wallet.append([]) # then we will have next a number of elements that corresponds to the number of stocks we are working with
        # inside then we will have elements arranged by entry price and volume of units we have bought of that stock with that entry price
        # this entry price will be used to calculate reward when selling occurs
        # e.g: wallet = [cash,[[entry_price1,volume1],[entry_price2,volume2]]]
        self.volume_history = [0]*self.window_size
        self.profit = 0
        self.stock_available = 0 # to include in the obs when wallet_in_obs == True
        self.stock_id = 1 #we still only have one stock in this env
        self.episode, self.max_eps_reward = self.get_episode()
        self.prices, self.signal_features = self._process_data()
        return self.get_observation()
        # in the new gym versions we need reset to have an dict element type to go with obs

    def step(self, action):
        self._done = False
        current_price = self.prices[self._current_tick]
        #print(action)
        v = action[1]
        # selecting the action correspondent to the index choosen by the agent that gives us a element [action.value,volume]
        # e.g: [1,3] = Buy 3 units
        # e.g: [0,3] = Sell 3 units
        step_reward = 0
        if v == 0:
            self.volume_history.append(0)
            #print('Hold')
            self._action_history.append(Actions.Hold.value)
        elif action[0] == Actions.Buy.value and self.wallet[0]>=v*current_price*(1+self.trade_fees): # having enough money to buy the selected volume
            #print('Buy')
            #print(v)
            self.Buy(action) # Buy function needs action element information to check the volume
            self._action_history.append(action[0]) # corresponds to one of the available Actions.values
        elif action[0] == Actions.Sell.value and self.stock_available>=v: # having enough units to satisfy asked volume
            # can't check the "have enoug volume condition" with len(self.wallet[stock_id])!=0 anymore, being that we can have stock units
            # but not enough to satisfy the asked volume
            #print('Sell')
            #print(v)
            step_reward=self.Sell(action)
            step_reward /= self.max_eps_reward
            self._total_reward += step_reward
            self._action_history.append(action[0]) # corresponds to one of the available Actions.values
        else:
            # action mistakes
            if v != 0: # to punish the agent when it chooses impossible actions for that step
                #print('mistake')
                step_reward = -self.penalty
                self._total_reward += step_reward
                self.num_punishments += 1 # to print on the graph
                self.volume_history.append('err')
                self._action_history.append(Actions.Hold.value) # corresponds to one of the available Actions.values
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            volume = self.volume_history[self._current_tick],
            action = action[0]
        )
        self._update_history(info)
        #sell and buy function need to check the current price, so we need to increment the time step after they are called
        if self._current_tick == self._end_tick:
            self._done = True
            self.num_punishments_history.append(self.num_punishments)
            self.profit = self.wallet[0]/self.initCash
            if self.mode == 'evaluate':
                self.render()
                if len(self.num_punishments_history) == self.n_eval_eps:
                    self.render_punish_hist()
                    self.mean_punish.append(np.mean(self.num_punishments_history))
                    self.render_mean_punish_hist()
                    self.num_punishments_history = []
        self._current_tick += 1
        return observation, step_reward, self._done, info
        #return observation, step_reward, self._done, False, info
        #(obs, rew, terminated, truncated, info), new gym api, all episodes are complete so truncated is always false

    def Buy(self,action):
        current_price = self.prices[self._current_tick]
        v = action[1] # volume
        self.wallet[0] -= current_price*v*(1+self.trade_fees) 
        same_entry_price_in_wallet = False 
        # we are going to check if we already have an element with the same entry price to each we can only sum the volume
        for i in range(len(self.wallet[self.stock_id])):
            if self.wallet[self.stock_id][i][1] == current_price:
                self.wallet[self.stock_id][i][0]+=v
                same_entry_price_in_wallet = True
                break
        if same_entry_price_in_wallet == False:
            self.wallet[self.stock_id].append([v,current_price]) # [volume_associated_with_entry_price_n_stock, entry_price]
        self.stock_available += v # self.stock_id-1 because id starts at 1 but list index starts at 0
        self.volume_history.append(v) 

    def Sell(self, action):
        current_price = self.prices[self._current_tick]
        v = action[1] # volume 
        self.volume_history.append(-v)
        step_reward = 0
        vol_counter = 0
        self.stock_available -= v
        while vol_counter < v:
            price = 0
            max_price_index = 0
            # we will discover the element with the max entry price to deal with the elements that give us less reward first and 
            # thus avoiding having stranded elements on the wallet and making the agent take responsabilitie for its bad buys
            for i in range(len(self.wallet[self.stock_id])):
                if self.wallet[self.stock_id][i][1] > price:
                    price = self.wallet[self.stock_id][i][1]
                    max_price_index = i
            if v-vol_counter >= self.wallet[self.stock_id][max_price_index][0]: 
                # if the asked volume is more then this element has, we take what is available here, delete the element and go search
                # for the next element with the biggest entry price to take units from there
                vol_counter += self.wallet[self.stock_id][max_price_index][0]
                step_reward += (current_price - self.wallet[self.stock_id][max_price_index][1])*self.wallet[self.stock_id][max_price_index][0]*(1-self.trade_fees) # self.wallet[stock_id][max_price_index][0] - transact_cost
                del self.wallet[self.stock_id][max_price_index]
            else:
                v1 = v-vol_counter
                vol_counter += v1
                self.wallet[self.stock_id][max_price_index][0] -= v1
                step_reward += (current_price - self.wallet[self.stock_id][max_price_index][1])*v1*(1-self.trade_fees) # self.wallet[stock_id][max_price_index][0] - transact_cost
        self.wallet[0] += step_reward # update cash
        return step_reward
        
    def get_observation(self):
        if self.wallet_in_obs == False:
            obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        elif self.wallet_in_obs == True:
            obs = np.append(self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1],[[self.wallet[0],self.stock_available]],axis=0)
        obs = obs.flatten()
        obs = np.append(obs,self._end_tick-self._current_tick)
        return obs

    def _process_data(self):
        prices = self.episode
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices, signal_features 

    def get_episode(self):
        points = np.array([1,2,4,7,10,12,10,7,4,2])
        num_waves = 3
        len_episode = self.frame_bound[1]-self.frame_bound[0]+self.window_size
        waves = []
        for _ in range(num_waves):
            wave = []
            counter = np.random.randint(0,len(points)-1)
            for j in range(len_episode):
                noise = np.random.normal()*self.eps_noise_level
                wave.append(points[counter]+noise)
                counter+=1
                if counter == len(points):
                    counter = 0
            waves.append(wave)                   
        final_wave = np.array(waves[0])/10
        for z in range(1,len(waves)):
            final_wave += np.array(waves[z])/10
        # we want to estimate de max reward for that episode, for that we want to make the difference between maximum element post
        # minimum and that minimum
        max_eps_reward, max_index, min_index = self.get_max_reward(final_wave)
        return final_wave, max_eps_reward 
    
    def get_max_reward(self, final_wave):
        # we want to estimate de max reward for that episode, for that we want to make the difference between maximum element post
        # minimum and that minimum. Sometimes though, the minimum of the episode can be on the last element, or the maximum in the first
        # one, so we want to find out the effective maximum reward. We are assuming that the best choice for episode is finding the 
        # combination that gives the most reward. In the real world that is not possible, the main strategy will never be to invest everything
        # in two peaks.
        maxs_index = []
        mins_index = []
        for i in range(len(final_wave)):
            if i == 0:
                if final_wave[i+1]>=final_wave[i]:
                    mins_index.append(i)
                elif final_wave[i+1]<=final_wave[i]:
                    maxs_index.append(i)
            elif i == len(final_wave)-1:
                if final_wave[i-1]>=final_wave[i]:
                    mins_index.append(i)
                elif final_wave[i-1]<=final_wave[i]:
                    maxs_index.append(i)
            elif (final_wave[i-1] <= final_wave[i] and final_wave[i+1] < final_wave[i]) or (final_wave[i-1] < final_wave[i] and final_wave[i+1] <= final_wave[i]):
                 maxs_index.append(i)
            elif (final_wave[i-1] >= final_wave[i] and final_wave[i+1] > final_wave[i]) or (final_wave[i-1] > final_wave[i] and final_wave[i+1] >= final_wave[i]):
                 mins_index.append(i)
        max_eps_reward = 0
        for i in maxs_index:
            for j in mins_index:
                if i>j:
                   if final_wave[i]-final_wave[j]>max_eps_reward:
                      max_index = i
                      min_index = j
                      max_eps_reward = final_wave[i]-final_wave[j]
        return max_eps_reward, max_index, min_index
            

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):       
        window_ticks = np.arange(len(self._action_history))
        buy_ticks = []
        sell_ticks = []
        hold_ticks = []
        for i in range(self.window_size,len(self._action_history)):
            if self._action_history[i] == Actions.Buy.value:
                buy_ticks.append(i)
            elif self._action_history[i] == Actions.Sell.value:
                sell_ticks.append(i)
            elif self._action_history[i] == Actions.Hold.value:
                hold_ticks.append(i)
        #plt.cla()
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot()
        plt.plot(window_ticks,self.prices)
        plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        plt.plot(buy_ticks, self.prices[buy_ticks], 'go')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo')
        for i in range(len(self.prices)):
            if self.volume_history[i]!=0:
                ax.annotate(str(self.volume_history[i]),xy=(i,self.prices[i]),xytext=(0,5), textcoords='offset points', size=15)
        
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Mistakes: %.6f" % self.num_punishments + ' ~ ' +
            "Units Left: %.6f" % self.stock_available+ ' ~ ' +
            "Profit: %.6f" % self.profit
        )
    def render_punish_hist(self):
        fig = plt.figure(figsize=(15,6))
        episodes = np.arange(len(self.num_punishments_history))
        plt.plot(episodes,self.num_punishments_history)

    def render_mean_punish_hist(self):
        fig = plt.figure(figsize=(15,6))
        evals = np.arange(len(self.mean_punish))
        plt.plot(evals,self.mean_punish)

    def reset_punish_hist(self):
        self.num_punishments_history = []  
    
    def reset_mean_punish_hist(self):
        self.mean_punish = []
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()