import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

#possible actions per step for the only available stock
class Actions(Enum):
    Sell = 0
    Buy = 1
    #Hold = 2

#possible positions per step for the only available stock, positions change when an action different from before is done
class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class stockMEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound):

        self.seed()
        self.df = df
        self.window_size = window_size # look back steps, sliding window
        self.frame_bound = frame_bound # interval of the data where the training will happen
        self.prices, self.signal_features = self._process_data()
        self.nStocks = 1
        self.obs_shape = (window_size, self.signal_features.shape[1]) # e.g: [[price1,diff1],[price2,diff2],[price3,diff3]] where diff = prc diff

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64)

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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._action = Actions.Sell.value
        self._position_history = (self.window_size * [None]) + [self._position] #[None, None, None, <Positions.Short: 0>]
        self._action_history = (self.window_size * [None]) + [self._action] #[None, None, None, 2(<Actions.Hold: 2>)]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            # we just trade one action in this env, so we can only buy/sell after sell/buy
            trade = True
        '''
        else:
            action = Actions.Hold.value 
            #Hold is given in this env as a concept to the model to evaluate train disturbance more then a command option
        '''
        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._action_history.append(action)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def _get_observation(self):
        #getting price and difference for the current sliding window
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]  


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        buy_ticks = []
        sell_ticks = []
        hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._action_history[i] == Actions.Buy.value:
                buy_ticks.append(tick)
            elif self._action_history[i] == Actions.Sell.value:
                sell_ticks.append(tick)
            '''
            elif self._action_history[i] == Actions.Hold.value:
                hold_ticks.append(tick)
            '''
        plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        plt.plot(buy_ticks, self.prices[buy_ticks], 'go')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if action == Actions.Sell.value: #aqui já não fazemos pela posição, mas pela ação especifica de vender
                step_reward += price_diff

        return step_reward
