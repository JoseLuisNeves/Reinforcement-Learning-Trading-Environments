import gym
from gym import spaces
from gym.utils import seeding
import more_itertools
import itertools
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy
import json



class Actions(Enum): 
    Sell = 0
    Buy = 1
    Hold = 2
    # we are here finally introducing volume to the multistock env, so we will start to have use for the Hold command. 
    # however, in the get_possible_actions function we will not include Hold in the action_combos, Hold actions will be represented
    # by Buy or Sell actions with volume zero in order to decrease the complexity of the function

class Positions(Enum):
    Short = 0
    Long = 1
    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, stocks, window_size, frame_bound, initCash, max_transactions, penalty=0, trade_fees = 0, wallet_in_obs = False, mode = 'training'):

        self.seed()
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.trade_fees = trade_fees
        self.penalty = penalty
        self.mode = mode
        if self.mode == 'evaluate':
            self.n_eval_eps = 10 # to get a universal storage for the env even when reseting to enable runs comparison
            self.mean_punish = []
        self.wallet_in_obs =  wallet_in_obs
        self.frame_bound = frame_bound
        self.initCash = initCash 
        self.window_size = window_size
        self.max_transactions = max_transactions
        self.multi_prices, self.multi_signal_features = self._process_data()
        if wallet_in_obs==False:
            self.shape = (self.num_stocks*self.window_size*self.multi_signal_features[0].shape[1],) 
            # observation is still only including price features still
            # [[[price11,diff11],[price12,diff12],[price12,diff12]],[[price21,diff21],[price22,diff22],[price22,diff22]],[[price31,diff31],[price32,diff32],[price32,diff32]]].flatten()
            # for a window_size = 3 and num_stocks = 3
        elif wallet_in_obs == True:
            self.shape = (self.num_stocks*(self.window_size+1)*self.multi_signal_features[0].shape[1],) 
            # observation includes now wallet info: the cash available, the number of units we have on our portfolio of each stock
            # [[[cash,num_stock1],[price11,diff11],[price12,diff12],[price13,diff13]],[[cash,num_stock2],[price21,diff21],[price22,diff22],[price22,diff22]],[[cash,num_stock3],[price31,diff31],[price32,diff32],[price32,diff32]]].flatten()
            # for a window_size = 3 and num_stocks = 3

        # spaces
        self.action_space = spaces.MultiDiscrete([len(Actions)-1,self.max_transactions+1]*self.num_stocks)
        # [action_stock1, volume_stock1, action_stock2, volume_stock2,...]
        # tried using  [[len(Actions)-1,self.max_transactions+1]]*self.num_stocks because it gave a more organized structure but is
        # not supported by openai gym yet, check https://github.com/openai/universe-starter-agent/issues/75
        #self.action_space = spaces.Discrete(((len(Actions)-1)*(self.max_transactions+1))**(self.num_stocks))
        # Discrete value that served has index of a list with all possible actions was another possible solution
        # for each stock we have we will have the action space of the gym_flexbuyselln_v1:
        # ((len(Actions)-1)*self.max_transactions+1)*((len(Actions)-1)*self.max_transactions+1))*... = 
        # = ((len(Actions)-1)*self.max_transactions+1)**(self.num_stocks) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
        #self.possible_actions = self.get_possible_actions(self.num_stocks,self.max_transactions)
        # for each one of the elements there computed in the gym_multistock_v1 there will be a branching of elements proportional to the 
        # max_transactions parameter. Every time we make append off a new combination, like we did in the gym_multistock_v1, we will append 
        # all of the elements associated with variance of possible volumes to each element inside the combination(except to the hold elements, 
        # that don't have volume associated)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.multi_prices[0]) - 1
        self._done = None
        self._current_tick = None
        self.action_history = None
        self._total_reward = None
        self.history = None
        self.wallet = None
        self.volume_history = None
        self.stock_available = None
        self.num_punishments = None
        self.num_punishments_history = []
        self.wallet = None
        self.stock_available = None
        # this list will keep track of available quantities of each stock to include in the observation see if asked and check if we have enough volume to satisfy the ask
        # it makes it so that we don't have to go on doing for loops every time we want to know the volume available, because the units are divided by entry price

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self.action_history = [[Actions.Hold.value]*self.num_stocks]*self.window_size
        self.volume_history = [[0]*self.num_stocks]*self.window_size
        self._total_reward = 0.
        self.history = {}
        self.wallet = []
        self.num_punishments = 0
        self.wallet.append(self.initCash) # cash is the first element of the wallet
        for i in range(self.num_stocks):
            self.wallet.append([]) # then we will have next a number of elements that corresponds to the number of stocks we are working with
        # inside then we will have elements arranged by entry price and volume of units we have bought of that stock with that entry price
        # this entry price will be used to calculate reward when selling occurs
        # e.g: wallet = [cash,[[entry_price1,volume1],[entry_price2,volume2]]]
        self.stock_available = [0]*self.num_stocks # to include in the obs when wallet_in_obs == True
        return self.get_observation()

    def step(self, action):
        self._done = False
        vol_hist_elem = [0]*self.num_stocks
        action_hist_elem = [Actions.Hold.value]*self.num_stocks
        for stock_id in range(self.num_stocks):
            #print('action',action, 'stock_id', stock_id, 'index',stock_id*2) 
            #print('wallet',self.wallet,'stock_available',self.stock_available)
            current_price = self.multi_prices[stock_id][self._current_tick]
            action_stock = action[stock_id*2]
            v = action[stock_id*2+1] # volume
            step_reward = 0
            if v == 0:
                action_hist_elem[stock_id] = Actions.Hold.value # Hold happens when selecting volume zero
            elif action_stock == Actions.Buy.value and self.wallet[0]>=v*current_price*(1+self.trade_fees): # having enough money to buy the selected volume
                #print('Buy', v)
                self.Buy(action,stock_id) # Buy function needs action element information to check the volume
                action_hist_elem[stock_id] = Actions.Buy.value
                vol_hist_elem[stock_id] = v
                self.stock_available[stock_id] += v
                #print('new wallet',self.wallet,'new stock available', self.stock_available)
                
            elif action_stock == Actions.Sell.value and self.stock_available[stock_id]>=v: # having enough units to satisfy asked volume
                # can't check the "have enoug volume condition" with len(self.wallet[stock_id])!=0 anymore, being that we can have stock units
                # but not enough to satisfy the asked volume
                #print('Sell',v)
                step_reward=self.Sell(action,stock_id) # Sell function needs action element information to check the volume
                action_hist_elem[stock_id] = Actions.Sell.value
                vol_hist_elem[stock_id] = -v
                self.stock_available[stock_id] -= v
                self._total_reward += step_reward  
                #print('new wallet',self.wallet,'new stock available', self.stock_available)
            else:
                # action mistakes
                step_reward = -self.penalty
                self._total_reward += step_reward
                self.num_punishments += 1 # to print on the graph
                action_hist_elem[stock_id] = Actions.Hold.value # when the agent makes a mistake we force him to hold
                vol_hist_elem[stock_id] = 'err'
                    
        self.action_history.append(action_hist_elem)
        self.volume_history.append(vol_hist_elem)
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            volume = self.volume_history[self._current_tick],
            action = self.action_history[self._current_tick]
        )
        self._update_history(info)
        if self._current_tick == self._end_tick:
            self._done = True
            self.num_punishments_history.append(self.num_punishments)
            if self.mode == 'evaluate':
                self.render()
                if len(self.num_punishments_history) == self.n_eval_eps:
                    self.render_punish_hist()
                    self.mean_punish.append(np.mean(self.num_punishments_history))
                    self.render_mean_punish_hist()
                    self.num_punishments_history = []
        self._current_tick += 1
        return observation, step_reward, self._done, info 

    def Buy(self,action,stock_id):
        current_price = self.multi_prices[stock_id][self._current_tick]
        v = action[stock_id*2+1] # volume
        self.wallet[0] -= current_price*v*(1+self.trade_fees) 
        same_entry_price_in_wallet = False 
        # we are going to check if we already have an element with the same entry price to each we can only sum the volume
        for i in range(len(self.wallet[stock_id+1])):# the first element of the wallet is cash
            if self.wallet[stock_id+1][i][1] == current_price:
                self.wallet[stock_id+1][i][0] += v
                same_entry_price_in_wallet = True
                break
        if same_entry_price_in_wallet == False:
            self.wallet[stock_id+1].append([v,current_price])
    
    def Sell(self, action, stock_id):
        current_price = self.multi_prices[stock_id][self._current_tick]
        v = action[stock_id*2+1] # volume
        step_reward = 0
        vol_counter = 0
        while vol_counter < v:
            price = 0
            max_price_index = 0
            # we will discover the element with the max entry price to deal with the elements that give us less reward first and 
            # thus avoiding having stranded elements on the wallet and making the agent take responsabilitie for its bad buys
            for i in range(len(self.wallet[stock_id+1])):
                if self.wallet[stock_id+1][i][1] > price:
                    price = self.wallet[stock_id+1][i][1]
                    max_price_index = i
            #print('wallet', self.wallet)
            #print('stock available', self.stock_available)
            #print('stock_id', stock_id,self.wallet[stock_id+1])
            #print(self.wallet[stock_id+1][max_price_index][0])
            if v-vol_counter >= self.wallet[stock_id+1][max_price_index][0]: 
                # if the asked volume is more then this element has, we take what is available here, delete the element and go search
                # for the next element with the biggest entry price to take units from there
                vol_counter += self.wallet[stock_id+1][max_price_index][0]
                step_reward += (current_price - self.wallet[stock_id+1][max_price_index][1])*self.wallet[stock_id+1][max_price_index][0]*(1-self.trade_fees) # self.wallet[stock_id][max_price_index][0] - transact_cost
                del self.wallet[stock_id+1][max_price_index]
            else:
                v1 = v-vol_counter
                vol_counter += v1
                self.wallet[stock_id+1][max_price_index][0] -= v1
                step_reward += (current_price - self.wallet[stock_id+1][max_price_index][1])*v1*(1-self.trade_fees) # self.wallet[stock_id][max_price_index][0] - transact_cost
        self.wallet[0] += step_reward # update cash
        return step_reward

    def get_observation(self):
        if self.wallet_in_obs == False:
            observation = []
            for stock_id in range(self.num_stocks):
                observation.append(self.multi_signal_features[stock_id][(self._current_tick-self.window_size+1):self._current_tick+1])
            return np.array(observation).flatten()
        elif self.wallet_in_obs == True:
            observation = []
            for stock_id in range(self.num_stocks):
                observation.append(self.multi_signal_features[stock_id][(self._current_tick-self.window_size+1):self._current_tick+1]+[[self.wallet[0],self.stock_available[stock_id]]])
            return np.array(observation).flatten()

    def get_possible_actions(self, num_stocks: int, max_transactions: int):
        file_path = '/content/drive/MyDrive/Tese_Verdadeira/buyselln_&_multistock/gym_stockflexbuyselln_multistock/gym_stockflexbuyselln_multistock/envs/'
        names_path = '/content/drive/MyDrive/Tese_Verdadeira/buyselln_&_multistock/gym_stockflexbuyselln_multistock/gym_stockflexbuyselln_multistock/envs/names.txt'
        new_name = str(num_stocks)+'_stocks_'+str(max_transactions)+'maxtrans'
        new_names_path = file_path + new_name
        names = []
        with open(names_path,'r') as file:
            lines = file.readlines() 
            for line in lines:
                names.append(line)
        if new_name not in names:
            actions = (i for i in range(len(Actions)-1) for _ in range(num_stocks))
            volumes = (i for i in range(max_transactions+1) for _ in range(num_stocks))
            # N.B. will not be sorted, will create a tuple the size of the Cartesian product (i.e. n_actions*max_transactions*num_stocks**2) 
            possible_actions = tuple(more_itertools.distinct_permutations(itertools.product(actions, volumes), num_stocks), file, indent=2)
            with open(new_names_path, 'w') as file:
                json.dump(possible_actions)
            with open(new_names_path, 'a') as file:
                file.write(new_name)
                file.write('\n')
            return possible_actions
        else:
            with open(names_path, 'r') as file:
                return json.load(file)
        
    def _process_data(self):
        multi_prices = []
        multi_signal_features = []
        for i in range(self.num_stocks):
            prices = self.stocks[i].loc[:, 'Close'].to_numpy()[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
            multi_prices.append(prices)
            #cortar a janela
            diff = np.insert(np.diff(prices), 0, 0)
            signal_features = np.column_stack((prices, diff))
            multi_signal_features.append(signal_features)
        return multi_prices, multi_signal_features 

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):
        
        window_ticks = np.arange(len(self.action_history))
        stocks_buy_ticks = []
        stocks_sell_ticks = []
        stocks_hold_ticks = []
        for stock_id in range(self.num_stocks):
            buy_ticks = []
            sell_ticks = []
            hold_ticks = []
            for j in range(self.window_size,len(self.action_history)):
                if self.action_history[j][stock_id] == Actions.Buy.value:
                    buy_ticks.append(j)
                elif self.action_history[j][stock_id] == Actions.Sell.value:
                    sell_ticks.append(j)
                elif self.action_history[j][stock_id] == Actions.Hold.value:
                    hold_ticks.append(j)
            stocks_buy_ticks.append(buy_ticks)
            stocks_sell_ticks.append(sell_ticks)
            stocks_hold_ticks.append(hold_ticks)
        #plt.cla()
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot()
        for stock_id in range(self.num_stocks):
            plt.plot(window_ticks,self.multi_prices[stock_id])
            plt.plot(stocks_sell_ticks[stock_id], self.multi_prices[stock_id][stocks_sell_ticks[stock_id]], 'ro')
            plt.plot(stocks_buy_ticks[stock_id], self.multi_prices[stock_id][stocks_buy_ticks[stock_id]], 'go')
            plt.plot(stocks_hold_ticks[stock_id], self.multi_prices[stock_id][stocks_hold_ticks[stock_id]], 'bo')
            for j in range(len(self.multi_prices[stock_id])):
                if self.volume_history[j][stock_id]!=0:
                    ax.annotate(str(self.volume_history[j][stock_id]),xy=(j,self.multi_prices[stock_id][j]), xytext=(0,5), textcoords='offset points', size=15)
        
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Mistakes: %.6f" % self.num_punishments + ' ~ ' +
            "Units Left: %.6f" % np.sum(np.array(self.stock_available))
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