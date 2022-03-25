import pandas as pd
import numpy as np
import gym
from gym import spaces
import random
from stable_baselines3.common.env_checker import check_env
from abides.agent.TradingAgent import TradingAgent
from .kernel_generator import kernel_generator
import _pickle as cpickle
from os.path import exists

class TraderEnv(gym.Env):
    """
    params:
    target: direction or price
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, name, columns_to_env, columns_to_scale, scaler_path, act_price, target, random_reset, apply_penalty = False):

        super(TraderEnv, self).__init__()
        print('###################################################### This is TraderEnv environment!')
        self.api = APIAgent(id ,name , type, symbol = None, starting_cash = None,)
        self.df.datetime = pd.to_datetime(self.df.datetime)
        self.df.index = range(len(self.df))
        self.target = target
        self.constant = 1
        self.stocks_list = list(self.df.stock.unique())

        self.apply_penalty = apply_penalty
        print(f'####### apply_penalty = {self.apply_penalty}')

        self.prev_action = 'No_Action'
        self.prev_trade_signal = 'No_Signal'
        self.total_nb_episods = len(self.df[self.df.end_episod == 1])

        self.columns_to_env = columns_to_env
        self.columns_to_scale = columns_to_scale
        if exists(scaler_path):
            with open(self.path2scaler, "rb") as input_file:
                self.scaler = cpickle.load(input_file)
        else:
            self.scaler = None

        self.act_price = act_price

        self.cols_to_report = ['index', 'stock', 'datetime', 'action', 'act_profit', self.act_price,
                               'reward', 'done', 'end_episod']


        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(len(self.columns_to_env) + 1 ,),
                                            dtype=np.float32)
        # self.current_stock_data = self.df.loc[:, self.columns_to_env].values.astype('float32')

        self.action_reward_df = None
        self.info2output = []
        #         self.gym_env_checker(nb_trials=100)
        self.prev_stg_act_profit = 0
        self.total_profit = 0
        self.reward = 0
        self.start_pos = self.current_index
        self.kernel = None
        self.data = []
    def reset(self):
        self.kernel = kernel_generator()
        self.data = []
        while len(self.data) < 50:
            self.data
        return self._observation()

    @staticmethod
    def action_type(x):
        if x in ['No_Action', 1]:
            return 'hold'
        elif x == 0:
            return 'Buy'
        elif x == 2:
            return 'Sell'

    @staticmethod
    def aux_signe(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0



    def step(self, action):
        # action: hold = 1  sell = 2 , buy = 0


        self.new_take_action(action)
        # self.stg_act_profit *= 3/2

        # self.reward =
        penalty = self.aux_penalty(action, self.trade_duration)

        self.reward = self.stg_act_profit

        self.start_pos += 1
        if action == self.prev_action:
            self.trade_duration += 1
        else:
            self.trade_duration = 1

        act = action

        row2append = row2append +  [self.current_stock, self.df.loc[self.current_index, 'datetime'],
                                   self.action_type(act), self.stg_act_profit,
                                   self.current_act_price, self.reward,
                                   self.done, self.df.loc[self.current_index, 'end_episod']]

        row2append += [self.constant]

        self.info2output.append(row2append)

        if self.done:

            if self.random_reset:
                self.current_index = random.randrange(len(self.df))
            else:
                self.current_index += 1
                if self.current_index >= len(self.df) - 1:
                    self.current_index = 0
            self.total_profit = 0
            self.reward = 0

            self.start_pos = 0
        else:
            self.current_index += 1
            if self.current_index >= len(self.df) - 1:
                self.current_index = 0

        obs = self._observation()
        # print(obs)
        self.prev_action = action
        self.prev_trade_signal = self.trade_signal
        self.prev_stg_act_profit = self.stg_act_profit
        self.done = not self.kernel.runner()
        return obs, self.reward, self.done, {}

    def _observation(self):
        # print(self.current_stock_data[self.current_index])
        current_stk_data = self.current_stock_data[self.current_index]
        current_stk_data = np.append(current_stk_data, [self.constant, (self.trade_duration - 1) / (330 - 1)])
        return current_stk_data


    def new_take_action(self, action):
        # action = 0: buy-long
        # action = 1: no-action
        # action = 2: sell-short

        self.current_act_price = self.df.loc[self.current_index, 'nscld_' + self.act_price]
        self.next_act_price = self.df.loc[self.current_index, 'next_' + self.act_price]
        self.current_stock = self.df.loc[self.current_index, 'stock']
        if self.done:
            self.stg_act_profit = 0
        else:

            if action == 0:
                self.stg_act_profit = 100 * (self.next_act_price - self.current_act_price) / self.current_act_price
            elif action == 2:
                self.stg_act_profit = 100 * (self.current_act_price - self.next_act_price) / self.next_act_price

            else:
                self.stg_act_profit = 0

    def _take_action(self, action):
        # action = 0: buy-long
        # action = 1: no-action
        # action = 2: sell-short

        self.current_act_price = self.df.loc[self.current_index, 'nscld_' + self.act_price]
        self.current_stock = self.df.loc[self.current_index, 'stock']

        if action == 0:
            if self.prev_action == 2:
                self.enter_long_act_price = self.current_act_price
                self.exit_short_act_price = self.current_act_price
                self.stg_act_profit = 100 * (
                            self.enter_short_act_price - self.exit_short_act_price) / self.exit_short_act_price
            elif self.prev_action in ['No_Action', 1]:
                self.enter_long_act_price = self.current_act_price
                self.stg_act_profit = 0
            else:
                self.stg_act_profit = 0


        elif action == 2:
            if self.prev_action == 0:
                self.enter_short_act_price = self.current_act_price
                self.exit_long_act_price = self.current_act_price
                self.stg_act_profit = 100 * (
                            self.exit_long_act_price - self.enter_long_act_price) / self.enter_long_act_price
            elif self.prev_action in ['No_Action', 1]:
                self.enter_short_act_price = self.current_act_price
                self.stg_act_profit = 0
            else:
                self.stg_act_profit = 0

        else:
            if self.prev_action == 0:
                self.exit_long_act_price = self.current_act_price
                self.stg_act_profit = 100 * (
                            self.exit_long_act_price - self.enter_long_act_price) / self.enter_long_act_price
            elif self.prev_action == 2:
                self.exit_short_act_price = self.current_act_price
                self.stg_act_profit = 100 * (
                            self.enter_short_act_price - self.exit_short_act_price) / self.exit_short_act_price
            else:
                self.stg_act_profit = 0

    def render(self, mode='human', close=False):
        pass

    def summary_action_reward_df(self):

        colms = ['stock', 'datetime', 'done', 'end_episod', 'action', self.act_price, 'act_profit', 'reward']
        self.action_reward_df = pd.DataFrame(self.info2output, columns=self.cols_to_report)
        return self.action_reward_df.loc[:, colms]

    def close(self):
        pass


class APIAgent(TradingAgent, TraderEnv):

  def __init__(self, id, name, type, symbol=None, starting_cash=None, within=0.01,
                random_state = None):
    # Base class init.
    super().__init__(id, name, type, starting_cash = starting_cash, random_state = random_state)

    self.symbol = symbol    # symbol to trade
    self.trading = False    # ready to trade
    self.traded = False     # has made its one trade
    # The amount of available "nearby" liquidity to consume when placing its order.
    # self.greed = greed      # trade this proportion of liquidity
    self.within = within    # within this range of the inside price


    # The agent begins in its "complete" state, not waiting for
    # any special event or condition.
    self.state = 'AWAITING_WAKEUP'



  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    super().wakeup(currentTime)

    if not self.mkt_open or not self.mkt_close:
      # TradingAgent handles discovery of exchange times.
      return
    else:
      if not self.trading:
        self.trading = True

        # Time to start trading!
        print ("{} is ready to start trading now.".format(self.name))


    # Steady state wakeup behavior starts here.

    # First, see if we have received a MKT_CLOSED message for the day.  If so,
    # there's nothing to do except clean-up.
    if self.mkt_closed and (self.symbol in self.daily_close_price):
      # Market is closed and we already got the daily close price.
      return


    self.setWakeup(currentTime + pd.Timedelta('1m'))


    # If the market is closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    if self.mkt_closed and (not self.symbol in self.daily_close_price):
      self.getLastTrade()
      self.state = 'AWAITING_LAST_TRADE'
      return


    # The impact agent will place one order based on the current spread.
    self.getCurrentSpread()
    self.state = 'AWAITING_SPREAD'

  # Request the last trade price for our symbol.
  def getLastTrade (self):
    super().getLastTrade(self.symbol)


  # Request the spread for our symbol.
  def getCurrentSpread (self, depth=10):
    # Impact agent gets depth 10000 on each side (probably everything).
    super().getCurrentSpread(self.symbol, depth)


  def getWakeFrequency (self):
    return (pd.Timedelta('1ns'))
