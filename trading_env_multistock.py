import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy


class Actions(Enum): 
    Sell = 0
    Buy = 1
    #Hold = 2 # since for each stock we trade only one unit we don't need the Hold command because the last_trade_tick covers that function

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, stocks, window_size, frame_bound, mode = 'train'):

        self.seed()
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.mode = mode
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.multi_prices, self.multi_signal_features = self._process_data()
        # in this env we introduce the possibility for a certain number of stocks for the first time, turning the observation to the shape
        # [[[price11,diff11],[price12,diff12],[price12,diff12]],[[price21,diff21],[price22,diff22],[price22,diff22]],[[price31,diff31],[price32,diff32],[price32,diff32]]].flatten()
        # for a window_size = 3 and num_stocks = 3
        self.shape = (self.num_stocks*self.window_size*self.multi_signal_features[0].shape[1],)
        # flatten is done by recomendation from the openai gym documentation

        # spaces
        self.action_space = spaces.Discrete(len(Actions)*self.num_stocks) 
        # In each step we choose one action for one step. The buy and sell processes are the same as in anytrading, just with multiple 
        # stocks and choosing a stock for the action, we are still just trading one unit of each stock
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.multi_prices[0]) - 1
        self._done = None
        self._current_tick = None
        self.last_trade_tick = None
        self._position = None
        self._position_history = None
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self.last_trade_tick = [self._current_tick - 1 for i in range(self.num_stocks)] 
        # we are still trading only one unit of each stock like anytrading here, so we are still going to use the last_trade_tick
        # for each stock to be certain that we only sell/buy after we buy/sell
        self._position = [Positions.Short for i in range(self.num_stocks)] 
        self._init_position = [None for i in range(self.num_stocks)] 
        self._position_history = self.window_size * [self._init_position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self.get_observation()


    def step(self, action_n_stock_id):
        self._done = False
        step_reward = self._calculate_reward(action_n_stock_id) 
        # contrary to the envs with flex volume or the v1 version of this env, instead of creating a list of possible actions 
        # we just have a range of number that have in them the volume and action information. E.g: 2, the number being even means the 
        # the choosen action is selling, 2/2 = 1, so that is the stock we are trading. Being PPO and A2C are models adapted to continuous 
        # action spaces this kinds of commands don't make much difference to them from the "getting_possible actions upon reset" method
        self._total_reward += step_reward

        if action_n_stock_id%len(Actions)==0:
            action = Actions.Sell.value # Sell
            stock_id = int(action_n_stock_id/len(Actions))

        else:
            action = Actions.Buy.value # Buy
            stock_id = int((action_n_stock_id-1)/len(Actions))

        trade = False
        if ((action == Actions.Buy.value) or
            (action == Actions.Sell.value)):
            trade = True
            if ((action == Actions.Buy.value and self._position[stock_id] == Positions.Short) or
            (action == Actions.Sell.value and self._position[stock_id] == Positions.Long)):
                self._position[stock_id] = self._position[stock_id].opposite()
                self.last_trade_tick[stock_id] = self._current_tick 
        position = deepcopy(self._position)
        self._position_history.append(position)
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position
        )
        #self._update_history(info)
        if self._current_tick == self._end_tick:
            self._done = True
            if self.mode == 'evaluate':
                self.render()
        self._current_tick += 1
        return observation, step_reward, self._done, info
        #return observation, step_reward, self._done, False, info
        #(obs, rew, terminated, truncated, info), new gym api, all episodes are complete so truncated is always false


    def get_observation(self):
        obs = []
        for i in range(self.num_stocks):
            obs.append(self.multi_signal_features[i][(self._current_tick-self.window_size+1):self._current_tick+1])
        obs = np.array(obs).flatten()
        return obs
    
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


    def _calculate_reward(self, action_n_stock_id):
        step_reward = 0
        if action_n_stock_id%len(Actions)==0:
            action = 0 # Sell
            stock_id = int(action_n_stock_id/len(Actions))
        else:
            action = 1 # Buy
            stock_id = int((action_n_stock_id-1)/len(Actions))

        trade = False

        if ((action == Actions.Buy.value and self._position[stock_id] == Positions.Short) or
            (action == Actions.Sell.value and self._position[stock_id] == Positions.Long)):
            trade = True

        if trade:
            current_price = self.multi_prices[stock_id][self._current_tick]
            last_trade_price = self.multi_prices[stock_id][self.last_trade_tick[stock_id]]
            price_diff = current_price - last_trade_price

            if action == Actions.Sell.value:
                #print('step reward')
                step_reward += price_diff

        return step_reward


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot()
        for i in range(self.num_stocks):
            plt.plot(self.multi_prices[i])
            short_ticks = []
            long_ticks = []
            window_ticks = np.arange(len(self._position_history))
            for j, tick in enumerate(window_ticks):
                if self._position_history[j][i] == Positions.Short:
                    short_ticks.append(tick)
                elif self._position_history[j][i] == Positions.Long:
                    long_ticks.append(tick)

            plt.plot(short_ticks, self.multi_prices[i][short_ticks], 'ro')
            plt.plot(long_ticks, self.multi_prices[i][long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()