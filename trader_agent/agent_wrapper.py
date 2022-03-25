import pandas as pd
import datetime as dt
from drl_agents.pre_processing_utils import Data_Preprocessing, Read_Fill_Missing, MakingTemaFeatures
from drl_agents import A2Cagent, PPOagent, DQNagent
import logging
import pytz
from copy import deepcopy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import torch as th
from env.environment import CustomEnv
import time

log = logging.getLogger(__name__)


class AgentWrapper():
    def __init__(self, stock, live_db_connection, model_dir, log_path="./log",
                 exp_name=None, rd_shuffle_seed=None, target='price', apply_penalty=applyPenalty,
                 use_strategy=useStrategy, random_reset=randomReset):
        log.info(f"Building agent wrapper for stock: {stock}")
        self.model_dir = model_dir
        self.stock = stock

        self.live_db_connection = live_db_connection
        self.exp_name = exp_name
        self.exp_time = dt.datetime.now(pytz.timezone("US/EASTERN")).strftime("%d-%b-%Y")
        self.version_date = dt.datetime.now(pytz.timezone("US/EASTERN")).strftime("%Y-%m-%d")
        self.log_path = log_path
        self.rd_shuffle_seed = rd_shuffle_seed
        self.target = target
        self.apply_penalty = apply_penalty
        self.use_strategy = use_strategy
        self.random_reset = random_reset

        log.info(f'use_strategy: {self.use_strategy}')
        log.info(f'apply_penalty: {self.apply_penalty}')
        log.info(f'random_reset: {self.random_reset}')

        self.path2scaler = os.path.join(self.model_dir, f"scaler_{self.stock}.pkl")
        self.path2scaledColumns = os.path.join(self.model_dir, f"scaledCols_{self.stock}.pkl")
        self.path2model = f"{self.model_dir}/model_{self.stock}"
        self.path2preds = f"{self.model_dir}/preds_{self.stock}"
        self.log_path = f"{self.model_dir}/evaluation_{self.stock}"

        self.stock_list = [self.stock]
        self.preproc_params = deepcopy(preproc_params)
        self.preproc_params.update({'stock': self.stock, 'rd_shuffle_seed': self.rd_shuffle_seed,
                                    'path2scaler': self.path2scaler, 'path2scaledColumns': self.path2scaledColumns})

        try:
            self.live_inferer = LiveInferer(self.stock, self.path2model, policy)
        except Exception as e:
            log.error(f"[AGENT WRAPPER] Can't use agent for inference due to: {str(e)}")



    def rename_raw_data(self, df, cols2keep=['datetime', 'stock', 'volume',
                                             'accumulated_volume', 'VWAP_per_candlestick',
                                             'open', 'close', 'high', 'low', 'VWAP']):
        if 'ticker' in df.columns:
            df = df.rename({'ticker': 'stock'}, axis=1)
        if 'posttime' in df.columns:
            df = df.drop(columns=['posttime'])
        if 'timestamp' in df.columns:
            df = df.rename({'timestamp': 'datetime'}, axis=1)
        df = df.loc[df.stock == self.stock]
        df = df.loc[:, cols2keep]
        df.datetime = pd.to_datetime(df.datetime)
        df = df.sort_values(by=['stock', 'datetime'], axis=0)
        df.index = range(len(df))
        return df

    def fill_missing_datetimes(self, df):
        RFM = Read_Fill_Missing(df)
        RFM.read_and_fill_missing_data()
        return RFM.df

    @staticmethod
    def add_tema_features(df):
        MTF = MakingTemaFeatures(df)
        df = MTF.from_raw_to_tema_features()
        return df

    def _train_pre_processing(self, df):
        df = df.loc[:, columns_to_pre_process]
        DP = Data_Preprocessing(df, **self.preproc_params)
        DP.data_to_taining_model()
        df, train, valid, test, self.scaler, self.columns_to_scale = DP.df, DP.train, DP.valid, DP.test, DP.scaler, DP.columns_to_scale

        test = test.sort_values(by='datetime')
        return df, train, valid, test

    def _inference_pre_processing(self, df):
        df = df.loc[:, columns_to_pre_process]
        DP = Data_Preprocessing(df, **self.preproc_params)
        DP.data_to_inference_model()
        df, train, valid, test, self.scaler, self.columns_to_scale = DP.df, DP.train, DP.valid, DP.test, DP.scaler, DP.columns_to_scale

        test = test.sort_values(by='datetime')
        #         test = test[test.date == test.date.unique()[-1]]
        ###Why the latest date? for inference in live but what for training-evaluation?
        return df, train, valid, test

    def update_env_params(self):
        train_env_params.update(
            {'scaler': self.scaler, 'columns_to_scale': self.columns_to_scale, 'target': self.target,
             'apply_penalty': self.apply_penalty, 'random_reset': self.random_reset})
        valid_env_params = deepcopy(train_env_params)
        valid_env_params['random_reset'] = False

        test_env_params = deepcopy(train_env_params)
        test_env_params['random_reset'] = False
        # print(th.__version__)
        nb_nodes = 64
        vf = [32]  # [64, 64, int(64//2)]
        pi = [256, 256, 256, 256]
        # act_fun = tanh # relu
        a2c_custom_params = {'policy_kwargs': dict(optimizer_class=RMSpropTFLike, activation_fn=th.nn.LeakyReLU,
                                                   net_arch=[dict(vf=vf, pi=pi)]),
                             'n_steps': 80, 'learning_rate': 0.00025, 'rms_prop_eps': 1e-05, 'use_rms_prop': True,
                             'normalize_advantage': True,
                             }
        a2c_params.update(a2c_custom_params)
        ppo_custom_params = {
            'policy_kwargs': dict(optimizer_class=RMSpropTFLike, activation_fn=th.nn.LeakyReLU, ortho_init=False,
                                  net_arch=[dict(vf=vf, pi=pi)]),
            'n_steps': 256, 'batch_size': 256, 'n_epochs': 10, 'learning_rate': 0.00035, 'clip_range': 0.5}

        ppo_params.update(ppo_custom_params)
        dqn_custom_params = {'policy_kwargs': dict(net_arch=pi, activation_fn=th.nn.LeakyReLU)}
        dqn_params.update(dqn_custom_params)

        early_stopping_params.update({'best_model_save_path': self.path2model, 'log_path': self.log_path})

        self.train_env_params, self.valid_env_params, self.test_env_params, self.a2c_params, self.ppo_params, self.early_stopping_params = (
        train_env_params,
        valid_env_params, test_env_params, a2c_params, ppo_params, early_stopping_params)

    def train(self, train, valid):

        if self.use_strategy:
            self.train_env = CustomEnv(train, **self.train_env_params)
            self.valid_env = CustomEnv(valid, **self.valid_env_params)


        self.train_env = make_vec_env(lambda: self.train_env, n_envs=1)
        self.nb_episods_in_valid = self.valid_env.total_nb_episods
        self.valid_env = make_vec_env(lambda: self.valid_env, n_envs=1)
        self.callbacks = []
        if use_early_stopping:
            self.early_stopping_params['n_eval_episodes'] = self.nb_episods_in_valid
            self.early_stopping = EarlyStoppingCallback(self.valid_env,
                                                        **self.early_stopping_params)  # EvalCallback(self.valid_env, **self.early_stopping_params)
            self.callbacks = [self.early_stopping]
        iteration = 0

        while True:
            iteration += 1

            if policy == 'a2c':
                self.a2c_agent = A2Cagent(train, self.train_env, policy='MlpPolicy', agt_params=self.a2c_params)
                # self.a2c_agent.Valid(,  early_stopping_params=  self.early_stopping_params)
                self.a2c_agent.learn(total_timesteps=20 * len(train), callback=self.callbacks)

            elif policy == 'ppo':
                self.ppo_agent = PPOagent(train, self.train_env, policy='MlpPolicy', agt_params=self.ppo_params)

                self.ppo_agent.learn(total_timesteps=20 * len(train), callback=self.callbacks)
            elif policy == 'dqn':
                self.dqn_agent = DQNagent(train, self.train_env, policy='MlpPolicy', agt_params=self.dqn_params)
                self.dqn_agent.learn(total_timesteps=20 * len(train), callback=self.callbacks)
            if self.early_stopping.best_mean_reward <= 0 and iteration < 2:
                self.early_stopping = EarlyStoppingCallback(self.valid_env,
                                                            **self.early_stopping_params)  # EvalCallback(self.valid_env, **self.early_stopping_params)
                self.callbacks = [self.early_stopping]
            else:
                break

    def predict(self, test, save_preds=True):
        LiveInf = LiveInferer(self.stock, self.path2model + '.zip', policy)
        test.index = range(len(test))
        LiveInf.infer(test, self.test_env_params)
        preds = LiveInf.preds
        if save_preds:
            preds.to_csv(self.path2preds, index=False)
        return preds

    def inference(self, test):
        date = test.date.unique()[-1]  ## Mostafa, TODO: remove this line if the operation is done elsewhere
        self.live_inferer.position = 'No_Signal'
        day_test = test[test.date == date]  ## Mostafa, TODO: remove this line
        day_test.reset_index(inplace=True)
        sm_test = day_test.iloc[-1:, :]
        sm_test.reset_index(inplace=True)
        latest_preds = self.live_inferer.infer(sm_test, self.test_env_params)
        latest_preds = latest_preds.iloc[-1:, :]
        latest_preds.drop(columns=["index"], inplace=True)
        return latest_preds

    def perform_train_prediction(self, df, save_preds=True, rtn=False):
        self.train_start_time = time.time()
        assert len(list((
                            df.datetime.dt.date).unique())) / 35 >= 1, '[AGENTWRAPPER-training] There is no enough data to perform the training.'
        self.pre_training_cleanup()
        df = self.rename_raw_data(df)
        df = self.fill_missing_datetimes(df)
        df = self.add_tema_features(df)

        df, train, valid, test = self._train_pre_processing(df)

        self.update_env_params()
        self.do_train(train, valid)
        preds = self.do_predict(test, save_preds=save_preds)
        self.post_training_cleanup()
        self.save_to_minio()
        if self.exp_name != "" and mlflow_available:
            try:
                self.record_mlflow_run(
                    params={"test_param": 1},
                    metrics={
                        "time_to_train": int(time.time() - self.train_start_time)
                    }
                )
            except Exception as e:
                log.error("[MLFLOW RECORD]", e)
                pass

        if rtn:
            return preds

    def perform_inference(self, df, store_in_table=None, send_to_topic=None):
        tt = time.time()
        log.info(f"Performing inference agent for stock: {self.stock}")
        self.load_from_minio()
        assert len(df) >= 1, '[AGENTWRAPPER-inference] There is no data to do the inference.'

        df = self.rename_raw_data(df)
        log.info(f"[{self.stock}] Renamed raw data at: {time.time() - tt}")
        df = self.add_tema_features(df)
        log.info(f"[{self.stock}] Added tema at: {time.time() - tt}")
        df, _, _, test = self._inference_pre_processing(df)
        log.info(f"[{self.stock}] Preprocessed at: {time.time() - tt}")
        self.update_env_params()
        log.info(f"[{self.stock}] Added tema at: {time.time() - tt}")

        latest_preds = self.do_inference(test)
        log.info(f"[{self.stock}] Predicted at: {time.time() - tt}")
        self.handle_outputs(latest_preds, store_in_table, send_to_topic)
        log.info(f"[{self.stock}] Handled at: {time.time() - tt}")
        return latest_preds

    def handle_outputs(self, preds, store_in_table=None, send_to_topic=None,
                       send_to_test_topic="i2_rl_all_signals_test"):
        if store_in_table is not None:
            to_sql_columns = ["stock", "datetime", "action", "trade_signal", "close"]
            preds[to_sql_columns].to_sql(con=self.live_db_connection, name=store_in_table, if_exists='append',
                                         index=False)
        if send_to_topic is not None:
            preds_dict = preds.to_dict(orient="records")
            if len(preds_dict) > 0:
                self.send_to_kafka(preds_dict[0], send_to_topic)
        if send_to_test_topic is not None:
            preds_dict = preds.to_dict(orient="records")
            if len(preds_dict) > 0:
                self.send_to_kafka(preds_dict[0], send_to_test_topic)
        return

