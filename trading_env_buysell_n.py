import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum): 
    Sell = 0
    Buy = 1
    #Hold = 2 #giving the hold option is unavoidable in this case because we want to distinguish volume

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound, initCash, mode = 'train'):

        self.seed()
        self.df = df
        self.window_size = window_size
        self.mode = mode
        self.frame_bound = frame_bound
        self.initCash = initCash # introducing money for the first time in this env
        self.prices, self.signal_features = self._process_data()
        self.num_stocks = 1
        self.obs_shape = (window_size*self.signal_features.shape[1],) 
        # flatten is done by recomendation from the openai gym documentation
        # obs = [[price_tminus2,price_diff_tminus3_&_tminus2],[price_tminus1,price_diff_tminus2_&_tminus1],[price_t,price_diff_tminus1_&_t]].flatten()
        # we still have only one stock to observe here so the shape is the same as anytrading and cmHold

        # spaces
        self.action_space = spaces.Discrete(len(Actions)) 
        # in this env the buy action will consist in buy every stock that the money permits in each step
        # the sell action will consist in selling every action we have
        # the results will be similar to previous envs, we want to explore how the notion of volume and money impacts the training
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64) # (continuous version)
        

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.wallet = None
        self.stock_id = None
        

    def check_wallet(self):
        return self.wallet

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._position = Positions.Short
        self._action = Actions.Hold.value
        self._position_history = [self._position]*self.window_size
        self._action_history = [self._action]*self.window_size
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.stock_id = 1
        self.wallet = []
        self.volume_history = [0]*self.window_size
        self.wallet.append(self.initCash) # cash is the first element of the wallet
        self.wallet.append([]) # then we will have next a number of elements that corresponds to the number of stocks we are working with
        # inside then we will have elements arranged by entry price and volume of units we have bought of that stock with that entry price
        # this entry price will be used to calculate reward when selling occurs
        # e.g: wallet = [cash,[[entry_price1,volume1],[entry_price2,volume2]]]
        return self.get_observation()
        # in the new gym versions we need reset to have an dict element type to go with obs

    def step(self, action):
        self._done = False
        current_price = self.prices[self._current_tick]
        v = 0
        step_reward = 0
        if ((action == Actions.Buy.value) or (action == Actions.Sell.value)):# we can buy/sell consecutively for the first time 
            if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)): 
            # we still play the change in position for visualization purposes
                self._position = self._position.opposite()
            	# we don't need last_trade_tick anymore
            if action == Actions.Buy.value and self.wallet[0]//current_price!=0:# checking if we have money to buy atleast one action
                v = self.Buy()
            elif action == Actions.Sell.value and len(self.wallet[self.stock_id])!=0:# checking if we have actions to sell
                step_reward , v = self.Sell()
                self._total_reward += step_reward  
            elif (action == Actions.Sell.value and len(self.wallet[self.stock_id])==0) or (action == Actions.Buy.value and self.wallet[0]//current_price==0):
                action = Actions.Hold.value 
                self.volume_history.append(0)
                # we don't penalize the agent when it fails to acknowledge the state of its wallet, 
                # it wouldn't make sense since we don't include it in the observation space
        else:
            self.volume_history.append(0)
        self._action_history.append(action)
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            volume = v,
            action = action
        )
        self._update_history(info)
        if self._current_tick == self._end_tick:
            self._done = True
            if self.mode == 'evaluate':
                self.render()
        self._current_tick += 1
        return observation, step_reward, self._done, info 
        # return observation, step_reward, self._done, False, info 
        #(obs, rew, terminated, truncated, info), new gym api, all episodes are complete so truncated is always false

    def Buy(self):
        current_price = self.prices[self._current_tick] 
        v = self.wallet[0]//current_price # this is volume, that will be as big as the current money(self.wallet[0][0]) allows
        self.volume_history.append(v)
        self.wallet[0]-= current_price*v 
        self.wallet[self.stock_id].append([v,current_price]) 
        # we save entry price to later calculate the reward from the difference between the then price and this one and then multiply
        # by the volume. We only get reward when we sell
        return v

    def Sell(self):
   		# selling everything!
        current_price = self.prices[self._current_tick]
        step_reward = 0
        v_total = 0
        for j in range(len(self.wallet[self.stock_id])): # len(self.wallet[i+1]) because the first element is the cash
                entry_price = self.wallet[self.stock_id][j][1]
                v = self.wallet[self.stock_id][j][0] #getting the volume from and entry price from each wallet element
                v_total += v
                step_reward += v*(current_price-entry_price) 
        self.volume_history.append(-v_total)
        #rebuilding the wallet
        self.wallet = []
        self.wallet.append(self.initCash) # cash is the first element of the wallet
        self.wallet.append([])
        return step_reward , v_total


    def get_observation(self):
        obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        obs = obs.flatten()
        return obs

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices, signal_features 


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
        print(self.prices)
        plt.plot(window_ticks,self.prices)
        plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        plt.plot(buy_ticks, self.prices[buy_ticks], 'go')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo')
        for i in range(len(self.prices)):
            if self.volume_history[i]!=0:
                ax.annotate(str(self.volume_history[i]),xy=(i,self.prices[i]),xytext=(0,5), textcoords='offset points', size=15)
        
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
        )
     
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()