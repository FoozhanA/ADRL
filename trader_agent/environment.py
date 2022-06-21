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
import time

class TraderEnv(gym.Env):
    """
    params:
    target: direction or price
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, apply_penalty = False, eval = -1):

        super(TraderEnv, self).__init__()
        print('###################################################### This is TraderEnv environment!')

        self.eval = eval
        # self.total_steps = 60 if eval else -1
        self.apply_penalty = apply_penalty
        # print(f'####### apply_penalty = {self.apply_penalty}')
        self.balance = 10000000 #100,000$ in cents

        self.scaler = MinMaxScaler()

        self.cols_to_report = ['symbol', 'timestamp', 'action', 'profit', 'close',
                               'reward']

        self.action_space = spaces.Box(low = -1, high = 1,shape = (1,))
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(27,),
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
        self.max_share = 50
        self.reward_coef = 2e-5
        self.order_book = None
        self.shares = 0
        self.reward = 0
        self.done = False
        self.total_profit = 0
        self.n_step = 0
        self.kernel = None
        # self.maxvec = [0 for i in range(27)]
        # self.minvec = [10000 for i in range(27)]
        # self.mnrew = [0 for i in range(2)]

        # minvec[
        #     108979, 555, 100506, 538, 100505, 434, 100500, 443, 100498, 395, 100497, 430, 100507, 1724, 100509, 1892, 100512, 3497, 100513, 3644, 100516, 3498, 4.633550612314139, 4.91304420213026, 96.02674239117559, 444.44444444444446, 80.95823671829234]
        # maxvec[
        #     -111034, 0, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, -6.405207242016331, -5.710141638034008, 3.947241014904165, -549.7076023392211, 6.6608465819867835]
        # minrew, maxrew[-11.1034, 10.8979]

        minvec = np.array([457951, 2403, 100506, 538, 100505, 522, 100500, 50000, 100498, 50001, 100497, 50001, 100507, 2019, 100509, 2583, 100512, 3497, 100513, 3644, 100526, 3498, 6.549383116827812, 5.595353704855847, 96.02674239117559, 481.48148148140666, 95.97485439875099])
        maxvec = np.array([ -482184, 0, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, -12.095042306216783, -12.640727662564942, 3.947241014904165, -549.7076023392211, 5.129108700382522])
        # minrew, maxrew[-48.2184, 45.795100000000005]
        self.state_coef = 1/np.maximum(abs(minvec), abs(maxvec))

    def reset(self):
        self.total_profit = 0
        self.shares = 0
        self.step_profit = 0
        self.kernel = kernel_generator()
        print('ran_reset in step: ', self.n_step)
        self.n_step = 0

        self.done = False
        self.balance = 10000000
        self.close_price = self.kernel.agents[0].order_books['ABM'].close
        self.order_book = self.kernel.agents[0].order_books['ABM']
        # self.kernel.runner()
        # self.close_price.append(self.kernel.agents[0].order_books['ABM'].close)
        # indicators = TEMA_indicators(self.close_price)
        # self.state = [self.balance] + [0] * self.depth * 2 + indicators

        # assertIs(self.order_book, self.kernel.agents[0].order_books['ABM'])

        return self._observation().astype(np.float32)

    def step(self, action):
        self.n_step += 1
        begin_asset = self.balance + self.shares * self.close_price[-1]

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

        self.state = self._observation()
        # self.maxvec = [max(self.maxvec[i], self.state[i]) for i in range(27)]
        # self.minvec = [min(self.minvec[i], self.state[i]) for i in range(27)]
        end_asset = self.balance + self.shares * self.close_price[-1]

        self.step_profit = end_asset - begin_asset

        self.total_profit = end_asset - 10000000
        self.reward = self.step_profit * self.reward_coef
        # print(self.reward,self.state)
        if self.eval != -1:
            # print('self.eval != -1 ', self.n_step)
            if self.n_step == self.eval:
                self.done = True
        if self.done:
            print(end_asset, self.total_profit)

        # self.mnrew[0], self.mnrew[1] = min(self.reward, self.mnrew[0]), max(self.reward, self.mnrew[1])
        return self.state, self.reward, self.done, {}

    def _observation(self):
        # TO-DO: change lists to np array
        # state = np.empty_like(self.observation_space)
        self.done = self.kernel.runner(intervals = 5)

        print('#############################', self.kernel.currentTime, self.done)
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
        ret = np.append([self.step_profit, self.shares] , [*bids , *asks , *indicators])
        return  ret * self.state_coef

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
