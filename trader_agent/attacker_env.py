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
from trader_agent.environment import TraderEnv
from stable_baselines3 import A2C

class AttackerEnv(TraderEnv):
    """
    params:
    target: direction or price
    """
    def __init__(self, trader_path = './best_model/best_model', apply_penalty = False, eval = -1):

        super(AttackerEnv, self).__init__(apply_penalty, eval)
        print('###################################################### This is AttackerEnv environment!')
        self.trader_path = trader_path
        self.trader_model = A2C.load(self.trader_path)
        self.trader_balance = 10000000
        self.trader_profit = 0
        self.trader_shares = 0
        self.dummy_kernel = None
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(29,),
                                            dtype=np.float32)
        self.trader_profit_1 = 0
        self.trader_profit_2 = 0
    def reset(self):
        self.trader_balance = 10000000
        self.trader_profit = 0
        self.trader_shares = 0
        self.trader_profit_1 = 0
        self.trader_profit_2 = 0
        self.state = super(AttackerEnv, self).reset()

        self.state += [self.trader_model.predict(self.state, deterministic=True)[0][0] * self.max_share, 0.0]
        # print(obs)
        return self.state

    def step(self, action):
        ##### add dummy vec
        self.n_step += 1
        begin_asset = self.shares * self.close_price[-1]
        trader_cost = 0
        trader_shares = 0
        trader_snd_cost = 0
        trader_snd_shares = 0
        trader_begin_asset = self.trader_balance + self.trader_shares * self.close_price[-1]
        # self.dummy_kernel = deepcopy(self.kernel)
        first_trader_action = self.trader_model.predict(self.state[:-2], deterministic=True)[0][0]

        is_buy = first_trader_action > 0
        trader_action = first_trader_action * self.max_share
        amount = int(abs(trader_action))

        if not is_buy:
            amount = min(amount, self.trader_shares)
        if amount > 0:
            trader_cost = amount * self.close_price[-1]
            trader_cost *= (-1 if not is_buy else 1)
            trader_shares = amount
            trader_shares *= -1 if not is_buy else 1

        # trader_end_asset = self.trader_balance + self.trader_shares * self.close_price[-1]
        # trader_first_profit = trader_end_asset - trader_begin_asset

        is_buy = action > 0
        action *= self.max_share
        amount = int(abs(action))
        if not is_buy:
            amount = min(amount, self.shares)
        if amount > 0:
            order = MarketOrder(0, self.kernel.agents[0].currentTime, 'ABM',amount , is_buy)
            cost, shares = self.kernel.agents[0].order_books['ABM'].handleMarketOrder(order)
            cost *= (-1 if not is_buy else 1)
            self.kernel.agents[0].publishOrderBookData()
            self.shares += shares * (-1 if not is_buy else 1)
            self.balance -= cost

        snd_trader_action = self.trader_model.predict(self._trader_observation(), deterministic=True)[0][0]
        if int(abs(first_trader_action)) != int(abs(snd_trader_action)):
            is_buy = snd_trader_action > 0
            trader_action = snd_trader_action * self.max_share
            amount = int(abs(trader_action))

            if not is_buy:
                amount = min(amount, self.trader_shares)
            if amount > 0:
                trader_snd_cost = amount * self.close_price[-1]
                trader_snd_cost *= (-1 if not is_buy else 1)
                trader_snd_shares = amount
                trader_snd_shares *= -1 if not is_buy else 1



        self.state = self._observation()


        end_asset = self.balance + self.shares * self.close_price[-1]
        step_profit = end_asset - begin_asset

        self.total_profit = end_asset - 10000000

        trader_profit_1 = trader_begin_asset + (-trader_cost + (self.trader_shares + trader_shares) * self.close_price[-1]) * -1
        trader_profit_2 = trader_begin_asset + (-trader_snd_cost + (self.trader_shares + trader_snd_shares) * \
                               self.close_price[-1])* -1

        self.trader_shares += trader_shares * (-1 if not is_buy else 1)
        self.trader_balance -= trader_cost
        self.trader_profit_1 += trader_profit_1
        self.trader_profit_2 += trader_profit_2

        self.reward = (step_profit + trader_profit_1 - trader_profit_2) * self.reward_coef + abs(first_trader_action - snd_trader_action)
        self.reward_change = abs(first_trader_action - snd_trader_action)
        self.state += [self.trader_model.predict(self.state, deterministic=True)[0][0] * self.max_share,
                       trader_profit_1 - trader_profit_2]
        if self.eval != -1 and self.n_step == self.eval:
            self.done = 1
        if self.done:
            print(self.n_step)
            print(end_asset, self.total_profit, self.trader_profit_2- self.trader_profit_1, (self.trader_profit_2 - self.trader_profit_1)/self.n_step , self.reward_change/self.n_step)
        return self.state, self.reward, self.done, {}

    def _observation(self):
        self.done = self.kernel.runner(intervals = 5)
        self.close_price = self.kernel.agents[0].order_books['ABM'].close

        indicators = TEMA_indicators(self.close_price)
        bids = self.order_book.getInsideBids(self.depth)
        if len(bids) < self.depth:
            bids.append((0,0,0))

        bids = [item for bid in bids for item in bid[:2]]
        asks = self.order_book.getInsideAsks(self.depth)

        if len(asks) < self.depth:
            asks.apped((0,0,0))
        asks = [item for ask in asks for item in ask[:2]]

        return [self.balance, self.shares] + bids + asks + indicators

    def _trader_observation(self, dummy=False):
        if dummy:
            self.dummy_kernel.runner(intervals=5)
        self.close_price = self.kernel.agents[0].order_books['ABM'].close
        indicators = TEMA_indicators(self.close_price)
        bids = self.order_book.getInsideBids(self.depth)
        if len(bids) < self.depth:
            bids.append((0,0,0))

        bids = [item for bid in bids for item in bid[:2]]
        asks = self.order_book.getInsideAsks(self.depth)

        if len(asks) < self.depth:
            asks.apped((0,0,0))
        asks = [item for ask in asks for item in ask[:2]]

        return [self.balance, self.trader_shares] + bids + asks + indicators

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
