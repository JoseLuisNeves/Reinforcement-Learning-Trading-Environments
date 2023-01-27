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
        self.shape = (self.num_stocks*self.window_size*self.multi_signal_features[0].shape[1],)
        # flatten is done by recomendation from the openai gym documentation

        # spaces
        self.action_space = spaces.MultiDiscrete([len(Actions)]*self.num_stocks)
        #self.action_space = spaces.Discrete(len(Actions)**(self.num_stocks))
        # in the first version of this env we were only allowed to choose one action for one stock so that the action space was of the size
        # len(Actions)*self.num_stocks. Here we want to choose an action for each stock per step, so that we will have 
        # len(Actions)*len(Actions)*len(Actions)... = len(Actions)**(self.num_stocks)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
        #self.possible_actions = self.get_possible_actions(self.num_stocks) 
        # since we have much more possibilities for actions per step in this case we will have to use a more organized way of getting the
        # available actions. In the readme image associated with this even there is a scheme explaining how the possible combinations are
        # obtained. The current get_possible_actions() code has only the capacity for 6 stocks max. When it has more it will print an error.
        # that have the possibility of more stocks, depending of the quantity, we will need to add blocks of iterations to being able to get
        # all of the combinations

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.multi_prices[0]) - 1
        self._done = None
        self._current_tick = None
        self.last_trade_tick = None
        self._position = None
        self._position_history = None
        self._action_history = None
        self.total_reward = None
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
        self._position = [Positions.Short for i in range(self.num_stocks)] 
        self._init_position = [None for i in range(self.num_stocks)] 
        self._position_history = self.window_size * [self._init_position]
        self.total_reward = 0.
        self._total_profit = 1.  # unit
        self.history = {}
        return self.get_observation()


    def step(self, action):
        self._done = False
        #action = self.possible_actions[action_id] # E.g: action = [0,1,0,1,1] for num_stocks=5
        step_reward = self._calculate_reward(action)
        self.total_reward += step_reward
        for stock_id in range(self.num_stocks):
            if ((action[stock_id] == Actions.Buy.value and self._position[stock_id] == Positions.Short) or
                (action[stock_id] == Actions.Sell.value and self._position[stock_id] == Positions.Long)):
                self._position[stock_id] = self._position[stock_id].opposite()
                self.last_trade_tick[stock_id] = self._current_tick 
        position = deepcopy(self._position)
        self._position_history.append(position)
        observation = self.get_observation()
        info = dict(
            total_reward = self.total_reward,
            total_profit = self._total_profit,
            position = self._position
        )
        if self._current_tick == self._end_tick:
            self._done = True
            if self.mode == 'evaluate':
                self.render()
        self._current_tick += 1
        return observation, step_reward, self._done,info
        #return observation, step_reward, self._done, False, info
        #(obs, rew, terminated, truncated, info), new gym api, all episodes are complete so truncated is always false


    def get_observation(self):
        obs = []
        for i in range(self.num_stocks):
            obs.append(self.multi_signal_features[i][(self._current_tick-self.window_size+1):self._current_tick+1])
        obs = np.array(obs).flatten()
        return obs

    def get_possible_actions(self, num_stocks):
        # see the scheme on the image associated with this code, c is buy(from "comprar" in portuguese) and v is sell(from "vender" in portuguese)
        possible_actions = []
        action = [0 for i in range(num_stocks)]
        possible_actions.append(action)
        counter = 1
        counter2 = 1
        while counter<num_stocks+1:
            list_of_counters = [i for i in range(counter)]
            for j in range(num_stocks):# c clusters movements
                action = [0 for i in range(num_stocks)]
                for k in range(counter):# elements of the elements of the list
                    if list_of_counters[k]>num_stocks-1:# when we arrive at the end of the list we want to get the c back to the beggining
                        # if the still don't have all of our combinations for that c cluster
                        list_of_counters[k]-=num_stocks
                    action[list_of_counters[k]] = 1
                    list_of_counters[k]+=1
                if action not in possible_actions:
                    possible_actions.append(action)
                while counter2<counter-1 and counter>=3:#dealing with inner combinations, repeat this block to be able to have >6 stocks
                    # on this block we are getting combos like [c,v,c,v,v,c], that we cannot get with just moving the c clusters, so 
                    # we move v clusters inside to get all of the combos
                    list_of_counters2 = [i for i in range(counter2)] # we go from the second element to the second last one 
                    for z1 in range(num_stocks):
                        action_inner_combs = deepcopy(action)
                        for z2 in range(counter2):
                            if list_of_counters2[z2]>num_stocks-1:# qd dá a volta quer voltar ao inicio da lista
                               list_of_counters2[z2]-=num_stocks
                            action_inner_combs[list_of_counters2[z2]] = 0
                            list_of_counters2[z2]+=1
                        if action_inner_combs not in possible_actions:
                            possible_actions.append(action_inner_combs)
                    counter2+=1
                counter2=1
            counter+=1
        if len(possible_actions) == len(Actions)**(num_stocks):
            return possible_actions
        else:
            print('Error while getting possible actions')

    def _process_data(self):
        multi_prices = []
        multi_signal_features = []
        for i in range(self.num_stocks):
            prices = self.stocks[i].loc[:, 'Close'].to_numpy()[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
            multi_prices.append(prices)
            diff = np.insert(np.diff(prices), 0, 0)
            signal_features = np.column_stack((prices, diff))
            multi_signal_features.append(signal_features)
        return multi_prices, multi_signal_features 


    def _calculate_reward(self, action):
        step_reward = 0
        trade = False
        for stock_id in range(self.num_stocks):           
            if ((action[stock_id] == Actions.Buy.value and self._position[stock_id] == Positions.Short) or
                (action[stock_id] == Actions.Sell.value and self._position[stock_id] == Positions.Long)):
                trade = True

            if trade:
                current_price = self.multi_prices[stock_id][self._current_tick]
                last_trade_price = self.multi_prices[stock_id][self.last_trade_tick[stock_id]]
                price_diff = current_price - last_trade_price

                if action[stock_id] == Actions.Sell.value:
                   step_reward += price_diff
        return step_reward


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        stocks_short_ticks = []
        stocks_long_ticks = []
        for stock_id in range(self.num_stocks):
            short_ticks = []
            long_ticks = []
            for j in range(self.window_size,len(self._position_history)):
                if self._position_history[j][stock_id] == Positions.Long:
                    long_ticks.append(j)
                elif self._position_history[j][stock_id] == Positions.Short:
                    short_ticks.append(j)
            stocks_short_ticks.append(short_ticks)
            stocks_long_ticks.append(long_ticks)
            
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot()
        for stock_id in range(self.num_stocks):
            plt.plot(window_ticks, self.multi_prices[stock_id])
            plt.plot(stocks_short_ticks[stock_id], self.multi_prices[stock_id][stocks_short_ticks[stock_id]], 'ro')
            plt.plot(stocks_long_ticks[stock_id], self.multi_prices[stock_id][stocks_long_ticks[stock_id]], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward 
        )