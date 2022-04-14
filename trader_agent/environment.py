import pandas as pd
import numpy as np
import gym
from gym import spaces
import random
from stable_baselines3.common.env_checker import check_env
from abides.agent.TradingAgent import TradingAgent
from trader_agent.kernel_generator import kernel_generator
import _pickle as cpickle
from os.path import exists
from copy import deepcopy
from trader_agent.preprocess import TEMA_indicators
from sklearn.preprocessing import MinMaxScaler
from collections import deque
# from unittest import assertIs
from abides.util.order.MarketOrder import MarketOrder

class TraderEnv(gym.Env):
    """
    params:
    target: direction or price
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, apply_penalty = False):

        super(TraderEnv, self).__init__()
        print('###################################################### This is TraderEnv environment!')



        self.apply_penalty = apply_penalty
        print(f'####### apply_penalty = {self.apply_penalty}')
        self.balance = 10000000 #100,000$ in cents

        self.scaler = MinMaxScaler()

        # self.cols_to_report = ['index', 'stock', 'datetime', 'action', 'profit', self.act_price,
        #                        'reward', 'done', 'end_episod']

        self.action_space = spaces.Box(low = -1, high = 1,shape = (1,))
        self.observation_space = spaces.Box(low=-1, high=2, shape=(25 ,),
                                            dtype=np.float32)
        # self.current_stock_data = self.df.loc[:, self.columns_to_env].values.astype('float32')

        self.action_reward_df = None
        self.info2output = []
        #         self.gym_env_checker(nb_trials=100)
        self.reward = 0
        self.kernel = None
        self.state = []
        self.close_price = deque(maxlen=50)
        self.depth = 5
        self.max_share = 100
        self.reward_coef = 1e-4
        self.order_book = None
        self.shares = 0
        self.reward = 0
        self.done = False

    def reset(self):
        self.shares = 0
        self.kernel = kernel_generator()
        self.done = False
        self.balance = 10000000
        self.close_price = deque(maxlen=50)
        self.close_price.extend([100000]*50)
        self.order_book = self.kernel.agents[0].order_books['ABM']
        # self.kernel.runner()
        # self.close_price.append(self.kernel.agents[0].order_books['ABM'].close)
        # indicators = TEMA_indicators(self.close_price)
        # self.state = [self.balance] + [0] * self.depth * 2 + indicators

        # assertIs(self.order_book, self.kernel.agents[0].order_books['ABM'])

        return self._observation()

    def step(self, action):
        begin_asset = self.balance + self.shares * self.close_price[-1]
        if action != 0:

            is_buy = action > 0
            action *= self.max_share
            if action < 0:
                action = min(action, self.shares)
            order = MarketOrder(0, self.kernel.agents[0].currentTime, 'ABM', action, is_buy)
            cost, shares = self.order_book['ABM'].handleMarketOrder(order)
            cost *= (-1 if not is_buy else 1)
            self.kernel.agents[0].publishOrderBookData()
            self.shares += shares * (-1 if not is_buy else 1)
            self.balance -= cost

        obs = self._observation()
        end_asset = self.balance + self.shares * self.close_price[-1]

        self.reward = (end_asset - begin_asset) * self.reward_coef

        # self.new_take_action(action)
        # # self.stg_act_profit *= 3/2
        #
        # # self.reward =
        # penalty = self.aux_penalty(action, self.trade_duration)
        #
        # self.reward = self.stg_act_profit
        #
        # self.start_pos += 1
        # if action == self.prev_action:
        #     self.trade_duration += 1
        # else:
        #     self.trade_duration = 1
        #
        # act = action
        #
        # row2append = row2append +  [self.current_stock, self.df.loc[self.current_index, 'datetime'],
        #                            self.action_type(act), self.stg_act_profit,
        #                            self.current_act_price, self.reward,
        #                            self.done, self.df.loc[self.current_index, 'end_episod']]
        #
        # row2append += [self.constant]
        #
        # self.info2output.append(row2append)
        #
        # if self.done:
        #
        #     if self.random_reset:
        #         self.current_index = random.randrange(len(self.df))
        #     else:
        #         self.current_index += 1
        #         if self.current_index >= len(self.df) - 1:
        #             self.current_index = 0
        #     self.total_profit = 0
        #     self.reward = 0
        #
        #     self.start_pos = 0
        # else:
        #     self.current_index += 1
        #     if self.current_index >= len(self.df) - 1:
        #         self.current_index = 0
        #
        # obs = self._observation()
        # # print(obs)
        # self.prev_action = action
        # self.prev_trade_signal = self.trade_signal
        # self.prev_stg_act_profit = self.stg_act_profit

        return obs, self.reward, self.done, {}

    def _observation(self):
        self.done = self.kernel.runner()
        self.close_price.extend(self.kernel.agents[0].order_books['ABM'].close)
        indicators = TEMA_indicators(self.close_price)
        bids = self.order_book.getInsideBids(self.depth)
        if len(bids) < self.depth:
            bids.apped((0,0,0))
        bids = [item for bid in bids for item in bid[:2]]
        asks = self.order_book.getInsideAsks(self.depth)
        if len(asks) < self.depth:
            asks.apped((0,0,0))
        asks = [item for ask in asks for item in ask[:2]]
        self.state = [self.balance] + bids + asks + indicators
        return self.state

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
