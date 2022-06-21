import numpy as np
import pandas as pd
import datetime as dt
from abides.Kernel import Kernel
from abides.util import util
from abides.util.order import LimitOrder
from abides.util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from tqdm import tqdm
from abides.agent.ExchangeAgent import ExchangeAgent
from abides.agent.NoiseAgent import NoiseAgent
from abides.agent.ValueAgent import ValueAgent
from abides.agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from abides.agent.examples.MomentumAgent import MomentumAgent
from abides.agent.examples.ExampleExperimentalAgent import ExampleExperimentalAgentTemplate, ExampleExperimentalAgent
from abides.model.LatencyModel import LatencyModel
import os, queue, sys
from abides.message.Message import MessageType

from abides.util.util import log_print
sys.path.append('/home/ict520c/Documents/ADRL/abides')




class CustomKernel(Kernel):
    def __init__(self,kernel_name, random_state, agents, startTime=None, stopTime=None,
             num_simulations=1, defaultComputationDelay=1,
             defaultLatency=1, agentLatency=None, latencyNoise=[1.0],
             agentLatencyModel=None, skip_log=False,
             seed=None, oracle=None, log_dir=None, ):
        super().__init__(kernel_name, random_state)
        self.agents = agents
        self.startTime = startTime
        self.stopTime = stopTime
        self.num_simulations = num_simulations
        self.defaultComputationDelay = defaultComputationDelay
        if agentLatency is None:
            self.agentLatency = [[defaultLatency] * len(agents)] * len(agents)
        else:
            self.agentLatency = agentLatency
        self.defaultLatency = defaultLatency
        self.latencyNoise = latencyNoise
        self.agentLatencyModel = agentLatencyModel
        self.skip_log = skip_log
        self.seed = seed
        self.oracle = oracle
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = str(int(self.kernelWallClockStart.timestamp()))
        self.agentCurrentTimes = [self.startTime] * len(self.agents)

        self.agentComputationDelays = [self.defaultComputationDelay] * len(self.agents)

        self.currentAgentAdditionalDelay = 0
        self.prevtime = self.startTime + pd.Timedelta(9, unit='h') + pd.Timedelta(30,unit='m')

    def start(self):
        self.simulation_start_time = dt.datetime.now()
        print("Simulation Start Time: {}".format(self.simulation_start_time))
        log_print("Kernel started: {}", self.name)
        log_print("Simulation started!")


        log_print("Starting sim {}", self.name)
        log_print("\n--- Agent.kernelInitializing() ---")
        for agent in self.agents:
            agent.kernelInitializing(self)
        log_print("\n--- Agent.kernelStarting() ---")
        for agent in self.agents:
            agent.kernelStarting(self.startTime)
        self.currentTime = self.startTime
        log_print("\n--- Kernel Clock started ---")
        log_print("Kernel.currentTime is now {}", self.currentTime)

        # Start processing the Event Queue.
        log_print("\n--- Kernel Event Queue begins ---")
        log_print("Kernel will start processing messages.  Queue length: {}", len(self.messages.queue))

        # Track starting wall clock time and total message count for stats at the end.
        self.eventQueueWallClockStart = pd.Timestamp('now')
        self.ttl_messages = 0

    def end_sim(self):
        if self.messages.empty():
            log_print("\n--- Kernel Event Queue empty ---")

        if self.currentTime and (self.currentTime > self.stopTime):
            log_print("\n--- Kernel Stop Time surpassed ---")

        # Record wall clock stop time and elapsed time for stats at the end.
        eventQueueWallClockStop = pd.Timestamp('now')

        eventQueueWallClockElapsed = eventQueueWallClockStop - self.eventQueueWallClockStart

        # Event notification for kernel end (agents may communicate with
        # other agents, as all agents are still guaranteed to exist).
        # Agents should not destroy resources they may need to respond
        # to final communications from other agents.
        log_print("\n--- Agent.kernelStopping() ---")
        for agent in self.agents:
            agent.kernelStopping()

        # Event notification for kernel termination (agents should not
        # attempt communication with other agents, as order of termination
        # is unknown).  Agents should clean up all used resources as the
        # simulation program may not actually terminate if num_simulations > 1.
        log_print("\n--- Agent.kernelTerminating() ---")
        for agent in self.agents:
            agent.kernelTerminating()

        print("Event Queue elapsed: {}, messages: {}, messages per second: {:0.1f}".format(
            eventQueueWallClockElapsed, self.ttl_messages,
            self.ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, 's')))))
        log_print("Ending sim {}", self.name)

        # The Kernel adds a handful of custom state results for all simulations,
        # which configurations may use, print, log, or discard.
        self.custom_state['kernel_event_queue_elapsed_wallclock'] = eventQueueWallClockElapsed
        self.custom_state['kernel_slowest_agent_finish_time'] = max(self.agentCurrentTimes)

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernelTerminating, but the Kernel must write out the summary
        # log itself.
        self.writeSummaryLog()

        # This should perhaps be elsewhere, as it is explicitly financial, but it
        # is convenient to have a quick summary of the results for now.
        print("Mean ending value by agent type:")
        for a in self.meanResultByAgentType:
            value = self.meanResultByAgentType[a]
            count = self.agentCountByType[a]
            print("{}: {:d}".format(a, int(round(value / count))))

        print("Simulation ending!")

        simulation_end_time = dt.datetime.now()
        print("Simulation End Time: {}".format(simulation_end_time))
        print("Time taken to run simulation: {}".format(simulation_end_time - self.simulation_start_time))

        return self.custom_state

    def runner(self,intervals = 1, *args, **kwargs):



        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.



        while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
            # print(len(self.messages.queue))
            # Get the next message in timestamp order (delivery time) and extract it.




            self.currentTime, event = self.messages.get()



            msg_recipient, msg_type, msg = event


            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 100000 == 0:
                log_print("\n--- Simulation time: {}, messages processed: {}, wallclock elapsed: {} ---\n".format(
                    self.fmtTime(self.currentTime), self.ttl_messages, pd.Timestamp('now') - self.eventQueueWallClockStart))

            log_print("\n--- Kernel Event Queue pop ---")
            log_print("Kernel handling {} message for agent {} at time {}",
                      msg_type, msg_recipient, self.fmtTime(self.currentTime))

            self.ttl_messages += 1

            # In between messages, always reset the currentAgentAdditionalDelay.
            self.currentAgentAdditionalDelay = 0

            # Dispatch message to agent.
            if msg_type == MessageType.WAKEUP:

                # Who requested this wakeup call?
                agent = msg_recipient


                # Test to see if the agent is already in the future.  If so,
                # delay the wakeup until the agent can act again.
                if self.agentCurrentTimes[agent] > self.currentTime:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put((self.agentCurrentTimes[agent],
                                       (msg_recipient, msg_type, msg)))
                    log_print("Agent in future: wakeup requeued for {}",
                              self.fmtTime(self.agentCurrentTimes[agent]))
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agentCurrentTimes[agent] = self.currentTime

                # Wake the agent.
                self.agents[agent].wakeup(self.currentTime)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                              self.currentAgentAdditionalDelay)

                log_print("After wakeup return, agent {} delayed from {} to {}",
                          agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

            elif msg_type == MessageType.MESSAGE:

                # Who is receiving this message?
                agent = msg_recipient
                # if self.agents[agent].name == 'EXAMPLE_EXPERIMENTAL_AGENT':
                #     print(msg_type,msg,self.currentTime)
                #     if msg.body['msg'] == "ORDER_EXECUTED":
                #         print(msg.body['order'].fill_price)
                #     if msg.body['msg'] == 'MARKET_DATA':
                #         self.agents[agent].placeMarketOrder(1,False)
                #
                # if self.agents[agent].name == "EXCHANGE_AGENT":
                #     if msg.body['sender'] == 5128:
                #         print(msg_type,msg,self.currentTime)
                #         if msg.body['msg'] == 'LimitOrder':
                #             break


                # Test to see if the agent is already in the future.  If so,
                # delay the message until the agent can act again.
                if self.agentCurrentTimes[agent] > self.currentTime:
                    # Push the message back into the PQ with a new time.
                    self.messages.put((self.agentCurrentTimes[agent],
                                       (msg_recipient, msg_type, msg)))
                    log_print("Agent in future: message requeued for {}",
                              self.fmtTime(self.agentCurrentTimes[agent]))
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agentCurrentTimes[agent] = self.currentTime

                # Deliver the message.
                self.agents[agent].receiveMessage(self.currentTime, msg)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                              self.currentAgentAdditionalDelay)

                log_print("After receiveMessage return, agent {} delayed from {} to {}",
                          agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))
                if self.prevtime <= self.currentTime and self.currentTime - self.prevtime >= pd.Timedelta(intervals, unit='min'):
                    self.prevtime = self.currentTime
                    return False


            else:
                raise ValueError("Unknown message type found in queue",
                                 "currentTime:", self.currentTime,
                                 "messageType:", self.msg.type)

        else:
            print('time', self.currentTime)
            return True


def kernel_generator(Exchange_Agent = 1, POV_Market_Maker_Agent = 1, Value_Agents = 100,
                    Momentum_Agents = 25, Noise_Agents = 5000, seed=413,
                    ticker='ABM',
                    historical_date='20200603',
                    start_time=dt.datetime.strptime('09:30:00','%H:%M:%S'),
                    end_time=dt.datetime.strptime('16:00:00','%H:%M:%S'), verbose=0,
                    fund_vol=1e-8, experimental_agent=False, ea_short_window='2min', ea_long_window='5min'):

    log_dir = f'log/experimental_agent_demo_short_2min_long_5min_{seed}'
    system_name = "ABIDES: Agent-Based Interactive Discrete Event Simulation"

    print ("=" * len(system_name))
    print (system_name)
    print ("=" * len(system_name))
    print ()

#     rgs, remaining_args = parser.parse_known_args()

    # if config_help:
    #     parser.print_help()
    #     sys.exit()

#     log_dir = log_dir  # Requested log directory.
#     seed = seed  # Random seed specification on the command line.
    if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
    np.random.seed(seed)

    util.silent_mode = not verbose
    LimitOrder.silent_mode = not verbose

    exchange_log_orders = True
    log_orders = None
    book_freq = 0


    print("Configuration seed: {}\n".format(seed))
    ########################################################################################################################
    ############################################### AGENTS CONFIG ##########################################################

    # Historical date to simulate.
    historical_date = pd.to_datetime(historical_date)
    mkt_open = historical_date + pd.to_timedelta(start_time.strftime('%H:%M:%S'))
    mkt_close = historical_date + pd.to_timedelta(end_time.strftime('%H:%M:%S'))
    agent_count, agents, agent_types = 0, [], []

    # Hyperparameters
    symbol = ticker
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    r_bar = 1e5
    sigma_n = r_bar / 10
    kappa = 1.67e-15
    lambda_a = 7e-11

    # Oracle
    symbols = {symbol: {'r_bar': r_bar,
                        'kappa': 1.67e-16,
                        'sigma_s': 0,
                        'fund_vol': fund_vol,
                        'megashock_lambda_a': 2.77778e-18,
                        'megashock_mean': 1e3,
                        'megashock_var': 5e4,
                        'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

    # 1) Exchange Agent


    stream_history_length = 25000

    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 mkt_open=mkt_open,
                                 mkt_close=mkt_close,
                                 symbols=[symbol],
                                 log_orders=exchange_log_orders,
                                 pipeline_delay=0,
                                 computation_delay=0,
                                 stream_history=stream_history_length,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    # 2) Noise Agents
    num_noise = 5000
    noise_mkt_open = historical_date + pd.to_timedelta("09:30:00")  # These times needed for distribution of arrival times
                                                                    # of Noise Agents
    noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
    agents.extend([NoiseAgent(id=j,
                              name="NoiseAgent {}".format(j),
                              type="NoiseAgent",
                              symbol=symbol,
                              starting_cash=starting_cash,
                              wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                              log_orders=log_orders,
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                   for j in range(agent_count, agent_count + num_noise)])
    agent_count += num_noise
    agent_types.extend(['NoiseAgent'])

    # 3) Value Agents
    num_value = 100
    agents.extend([ValueAgent(id=j,
                              name="Value Agent {}".format(j),
                              type="ValueAgent",
                              symbol=symbol,
                              starting_cash=starting_cash,
                              sigma_n=sigma_n,
                              r_bar=r_bar,
                              kappa=kappa,
                              lambda_a=lambda_a,
                              log_orders=log_orders,
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                   for j in range(agent_count, agent_count + num_value)])
    agent_count += num_value
    agent_types.extend(['ValueAgent'])

    # 4) Market Maker Agents

    """
    window_size ==  Spread of market maker (in ticks) around the mid price
    pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
           the market maker places at each level
    num_ticks == Number of levels to place orders in around the spread
    wake_up_freq == How often the market maker wakes up

    """

    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_params = [('adaptive', 0.025, 10, '10S', 1),
                 ('adaptive', 0.025, 10, '10S', 1)
                 ]

    num_mm_agents = len(mm_params)
    mm_cancel_limit_delay = 50  # 50 nanoseconds

    agents.extend([AdaptiveMarketMakerAgent(id=j,
                                    name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                    type='AdaptivePOVMarketMakerAgent',
                                    symbol=symbol,
                                    starting_cash=starting_cash,
                                    pov=mm_params[idx][1],
                                    min_order_size=mm_params[idx][4],
                                    window_size=mm_params[idx][0],
                                    num_ticks=mm_params[idx][2],
                                    wake_up_freq=mm_params[idx][3],
                                    cancel_limit_delay=mm_cancel_limit_delay,
                                    skew_beta=0,
                                    level_spacing=5,
                                    spread_alpha=0.75,
                                    backstop_quantity=50000,
                                    log_orders=log_orders,
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                              dtype='uint64')))
                   for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
    agent_count += num_mm_agents
    agent_types.extend('POVMarketMakerAgent')


    # 5) Momentum Agents
    num_momentum_agents = 25

    agents.extend([MomentumAgent(id=j,
                                 name="MOMENTUM_AGENT_{}".format(j),
                                 type="MomentumAgent",
                                 symbol=symbol,
                                 starting_cash=starting_cash,
                                 min_size=1,
                                 max_size=10,
                                 wake_up_freq='20s',
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                           dtype='uint64')))
                   for j in range(agent_count, agent_count + num_momentum_agents)])
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")

    # 6) Experimental Agent

    ### Example Experimental Agent parameters

    # if experimental_agent:
    #     experimental_agent = ExampleExperimentalAgent(
    #     id=agent_count,
    #     name='EXAMPLE_EXPERIMENTAL_AGENT',
    #     type='ExampleExperimentalAgent',
    #     symbol=symbol,
    #     starting_cash=starting_cash,
    #     levels=5,
    #     subscription_freq=1e9,
    #     wake_freq='10s',
    #     order_size=100,
    #     short_window=ea_short_window,
    #     long_window=ea_long_window,
    #     log_orders=True,
    #     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
    #     )
    # else:
    #     experimental_agent = ExampleExperimentalAgentTemplate(
    #     id=agent_count,
    #     name='EXAMPLE_EXPERIMENTAL_AGENT',
    #     type='ExampleExperimentalAgent',
    #     symbol=symbol,
    #     starting_cash=starting_cash,
    #     levels=10,
    #     subscription_freq=60e9,
    #     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

    # if config_help:
    #     parser.print_help()
    #     sys.exit()


    # experimental_agents = [experimental_agent]
    # agents.extend(experimental_agents)
    # agent_types.extend("ExperimentalAgent")
    # agent_count += 1



    ########################################################################################################################
    ########################################### KERNEL AND OTHER CONFIG ####################################################



    kernelStartTime = historical_date

    kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

    defaultComputationDelay = 50  # 50 nanoseconds

    # LATENCY

    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
    pairwise = (agent_count, agent_count)

    # All agents sit on line from Seattle to NYC
    nyc_to_seattle_meters = 3866660
    pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                            random_state=latency_rstate)
    pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

    model_args = {
        'connected': True,
        'min_latency': pairwise_latencies
    }

    latency_model = LatencyModel(latency_model='deterministic',
                                 random_state=latency_rstate,
                                 kwargs=model_args
                                 )
    # KERNEL
    kernel = CustomKernel("RMSC03 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                           dtype='uint64')),
                            agents=agents,
                            startTime=kernelStartTime,
                            stopTime=kernelStopTime,
                            agentLatencyModel=latency_model,
                            defaultComputationDelay=defaultComputationDelay,
                            oracle=oracle,
                            log_dir=log_dir)

    kernel.start()
    return kernel
    # kernel.runner(agents=agents,
    #               startTime=kernelStartTime,
    #               stopTime=kernelStopTime,
    #               agentLatencyModel=latency_model,
    #               defaultComputationDelay=defaultComputationDelay,
    #               oracle=oracle,
    #               log_dir=log_dir)
