import pandas as pd
import numpy as np
import gym
from gym import spaces
import random
from stable_baselines3.common.env_checker import check_env





class StrategyEnv(gym.Env):
    """
    params:
    target: direction or price
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, columns_to_env, columns_to_scale, scaler, act_price, target, random_reset, apply_penalty):

        super(StrategyEnv, self).__init__()
        print('###################################################### This is strategy environment!')
        self.df = df
        self.df.datetime = pd.to_datetime(self.df.datetime)
        self.df.index = range(len(self.df))
        self.target = target
        self.constant = 1
        self.stocks_list = list(self.df.stock.unique())

        self.random_reset = random_reset

        if self.random_reset:
            print('episodes start randomly!')
            self.current_index = random.randrange(len(self.df))
        else:
            self.current_index = 0

        self.apply_penalty = apply_penalty
        print(f'####### apply_penalty = {self.apply_penalty}')

        self.prev_action = 'No_Action'
        self.prev_trade_signal = 'No_Signal'
        self.total_nb_episods = len(self.df[self.df.end_episod == 1])

        self.columns_to_env = columns_to_env
        self.columns_to_scale = columns_to_scale
        self.scaler = scaler
        self.act_price = act_price

        self.cols_to_report = ['index', 'stock', 'datetime', 'action', 'act_profit', self.act_price,
                               'reward', 'done', 'end_episod']

        feedback_cols = ['constant']
        self.cols_to_report += feedback_cols

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(len(self.columns_to_env) + 1 + len(feedback_cols),),
                                            dtype=np.float32)
        self.current_stock_data = self.df.loc[:, self.columns_to_env].values.astype('float32')

        self.action_reward_df = None
        self.info2output = []
        self.trade_duration = 1
        #         self.gym_env_checker(nb_trials=100)
        self.prev_stg_act_profit = 0
        self.total_profit = 0
        self.reward = 0
        self.start_pos = self.current_index

    def reset(self):
        self.prev_action = 'No_Action'
        self.prev_trade_signal = 'No_Signal'
        return self._observation()

    @staticmethod
    def action_type(x):
        if x in ['No_Action', 1]:
            return 'No_Action'
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

    def aux_penalty(self, x, duration):
        if self.apply_penalty:
            if x != self.prev_action and self.prev_action not in ['No_Action', 1]:
                return abs(self.stg_act_profit * (1 / (np.log(duration + 5) ** 2)))

        return 0

    def step(self, action):
        # action: hold = 1  sell = 2 , buy = 0
        self.done = (int(self.df.loc[self.current_index, 'end_episod'] == 1) == 1)
        row2append = [self.current_index]
        self.new_take_action(action)
        # self.stg_act_profit *= 3/2
        self.trade_signal = self._update_position(self.action_type(self.prev_action), self.action_type(action),
                                                  self.prev_trade_signal)
        # self.reward =
        penalty = self.aux_penalty(action, self.trade_duration)

        step_reward = self.stg_act_profit - penalty

        # if step_reward > 0:
        #     step_reward *= 1.5
        # self.total_profit += step_reward
        #
        #
        # self.reward = (self.total_profit) / (self.start_pos+1)
        self.reward = step_reward  # *2
        # normalize reward

        self.start_pos += 1
        if action == self.prev_action:
            self.trade_duration += 1
        else:
            self.trade_duration = 1

        act = action

        row2append = row2append + [self.current_stock, self.df.loc[self.current_index, 'datetime'],
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
        return obs, self.reward, self.done, {}

    def _observation(self):
        # print(self.current_stock_data[self.current_index])
        current_stk_data = self.current_stock_data[self.current_index]
        current_stk_data = np.append(current_stk_data, [self.constant, (self.trade_duration - 1) / (330 - 1)])
        return current_stk_data

    @staticmethod
    def _update_position(prev_act, act, prev_trade_signal):

        # action = 0: buy-long
        # action = 1: no-action
        # action = 2: sell-short

        trade_recommendation_dict = {'Enter Long': 'LONG BUY', 'Enter Short': 'SHORT SELL',
                                     'Exit Long': 'LONG SELL', 'Exit Short': 'SHORT BUY',
                                     'Exit Short , Enter Long': 'SHORT BUY , LONG BUY',
                                     'Exit Long , Enter Short': 'LONG SELL , SHORT SELL',
                                     'No_Signal': 'No_Signal', 'Stay Long': 'Stay Long',
                                     'Stay Short': 'Stay Short', }

        if prev_act == 'Buy':
            if act == 'Buy':
                trade_signal = 'Stay Long'
            elif act == 'Sell':
                trade_signal = 'Exit Long , Enter Short'
            else:
                trade_signal = 'Exit Long'

        elif prev_act == 'Sell':
            if act == 'Buy':
                trade_signal = 'Exit Short , Enter Long'
            elif act == 'Sell':
                trade_signal = 'Stay Short'
            else:
                trade_signal = 'Exit Short'
        elif prev_act == 'No_Action':
            if act == 'Buy':
                trade_signal = 'Enter Long'
            elif act == 'Sell':
                trade_signal = 'Enter Short'
            else:
                trade_signal = 'No_Signal'

        if trade_signal in trade_recommendation_dict.keys():
            new_trade_signal = trade_recommendation_dict[trade_signal]
        return new_trade_signal

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

    def gym_env_checker(self, nb_trials=200):
        kk = 0
        while kk <= nb_trials - 1:
            check_env(self, warn=False)
            kk += 1
        if kk == nb_trials:
            print('the environment is very likely to be compatible with the agent')

    def summary_action_reward_df(self):

        colms = ['stock', 'datetime', 'done', 'end_episod', 'action', self.act_price, 'act_profit', 'reward']
        self.action_reward_df = pd.DataFrame(self.info2output, columns=self.cols_to_report)
        return self.action_reward_df.loc[:, colms]

    def close(self):
        pass