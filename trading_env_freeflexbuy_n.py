import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from copy import deepcopy


class Actions(Enum): 
    Sell = 0
    Buy = 1
    Hold = 2 #giving the hold option is unavoidable in this case because we want to distinguish volume

class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound, initCash, max_transactions, mode = 'train', trade_fees = 0, wallet_in_obs = False, penalty = 0):

        self.seed()
        self.df = df
        self.penalty = penalty
        self.mode = mode
        if self.mode == 'evaluate':
            self.n_eval_eps = 10 # to get a universal storage for the env even when reseting to enable runs comparison
            self.mean_punish = []
        self.window_size = window_size
        self.trade_fees = trade_fees
        self.wallet_in_obs = wallet_in_obs
        self.max_transactions = max_transactions
        self.initCash = initCash
        self.frame_bound = frame_bound
        self.prices, self.signal_features = self._process_data()
        self.num_stocks = 1
        if self.wallet_in_obs == False:
            # obs = [[price_tminus2,price_diff_tminus3_&_tminus2],[price_tminus1,price_diff_tminus2_&_tminus1],[price_t,price_diff_tminus1_&_t]].flatten()
            self.obs_shape = (window_size*self.signal_features.shape[1],) 
        elif self.wallet_in_obs == True:
            # obs = [[cash,num_stocks_available],[price_tminus2,price_diff_tminus3_&_tminus2],[price_tminus1,price_diff_tminus2_&_tminus1],[price_t,price_diff_tminus1_&_t]].flatten()
            self.obs_shape = ((window_size+1)*self.signal_features.shape[1],)
        # flatten is done by recomendation from the openai gym documentation

        # spaces
        self.action_space = spaces.Discrete(len(Actions)+self.max_transactions-1)
        # in this env we want to give flexibility of volume to the buy action, so the action of buying diverges to a number of actions
        # equal to max_transactions. The total number of actions will be then that plus the action of selling and the action of holding:
        # len(Actions)+self.max_transactions-1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64)
        self.possible_actions = self.get_possible_actions()
        # this function builds the action space by getting every possible action for each step

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.volume_history = None
        self.wallet = None
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
        self._action = Actions.Hold.value # actions default to the first lookback points that are not include in frame_bound
        self._action_history = [self._action]*self.window_size
        self._total_reward = 0.
        self._total_profit = 1. # not usefull, episode reward include this info
        self._first_rendering = True
        self.history = {}
        self.stock_id = 1
        self.volume_history = [0]*self.window_size 
        self.wallet = []
        self.num_punishments = 0
        self.stock_available = 0
        self.wallet.append(self.initCash)
        self.wallet.append(self.stock_available) 
        # wallet structure is what is going to distinguish the flexbuy env from the freeflexbuy one.
        # we are going to have only volume on the stock element of the wallet, and the reward will be made at the end of the episode
        # by the difference between the initial cash and final cash
        # since the reward comes from selling and in this env we sell everything everytime we sell we will have a less specific reward
        # either way and see less difference between the envs
        return self.get_observation()
        # in the new gym versions we need reset to have an dict element type to go with obs


    def step(self, action_id):
        self._done = False
        action = deepcopy(self.possible_actions[action_id]) # to prevent change in possible_actions list when forcing hold in a mistake
        current_price = self.prices[self._current_tick]
        v = action[1]
        # selecting the action correspondent to the index choosen by the agent that gives us a element [action.value,volume]
        # e.g: [1,3] = Buy 3 units
        # note: the sell action only has the element [0,0], given which the agent sells everything it has
        step_reward = 0
        if action[0] == Actions.Buy.value and self.wallet[0]>=v*current_price*(1+self.trade_fees): # having enough money to buy the selected volume
            self.Buy(action)
        elif action[0] == Actions.Sell.value and self.wallet[self.stock_id]>0: # having at least one unit to sell
            step_reward=self.Sell()
            self._total_reward += step_reward
        else:
            # Hold+action mistakes
            if action[0] != Actions.Hold.value: # to punish the agent when it chooses impossible actions for that step
                step_reward = -self.penalty
                self._total_reward += step_reward
                self.num_punishments += 1 # to print on the graph
                self.volume_history.append('err')
                action[0] = Actions.Hold.value # when the agent makes a mistake we force him to hold
            else:
                self.volume_history.append(0)

        self._action_history.append(action[0])
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            volume = self.volume_history[self._current_tick],
            action = action[0]
        )
        self._update_history(info)
        #buy function needs to check the current price, so we need to increment the time step after its called
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
        #return observation, step_reward, self._done, False, info
        #(obs, rew, terminated, truncated, info), new gym api, all episodes are complete so truncated is always false

    def Buy(self, action):
        # we apply trade fees, when they present, only to this function in this env, because it is the one that
        # permits flexibility and, because of that, the one that corresponds to the affected action
        current_price = self.prices[self._current_tick]
        v = action[1]
        self.wallet[0] -= current_price*v*(1+self.trade_fees) 
        self.wallet[self.stock_id] += v
        self.volume_history.append(v)

    def Sell(self):
   		#selling everything we have
        current_price = self.prices[self._current_tick]
        v = self.wallet[self.stock_id]
        self.wallet[self.stock_id] = 0
        self.wallet[0] += v*current_price
        self.volume_history.append(-v)
        return v*current_price

    def get_possible_actions(self):
        possible_actions = []
        possible_actions.append([Actions.Sell.value,0])
        for j in range(1,self.max_transactions+1):#e.g:we don't include buying with volume zero, in future env's that will correspond to holding 
                possible_actions.append([Actions.Buy.value,j]) #[action,volume]
        possible_actions.append([Actions.Hold.value,0])
        return possible_actions
        

    def get_observation(self):
        if self.wallet_in_obs == False:
            obs = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        elif self.wallet_in_obs == True:
            obs = np.append(self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1],[[self.wallet[0],self.stock_available]],axis=0)
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
        plt.plot(window_ticks,self.prices)
        plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        plt.plot(buy_ticks, self.prices[buy_ticks], 'go')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo')
        for i in range(len(self.prices)):
            if self.volume_history[i]!=0:
                ax.annotate(str(self.volume_history[i]),xy=(i,self.prices[i]),xytext=(0,5), textcoords='offset points', size=15)
        
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward+ ' ~ ' +
            "Total Mistakes: %.6f" % self.num_punishments  + ' ~ ' +
            "Units Left: %.6f" % self.wallet[self.stock_id]
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