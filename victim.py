from abides.agent import TradingAgent

class DeepQLearning(TradingAgent):
    def __init__(self, id, name, type, symbol='IBM', random_state=None, starting_cash=100000, log_orders=False, log_to_file=True):
        super().__init__(id, name, type, random_state, log_to_file)
        self.symbol = symbol

    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        if self.state == 'AWAITING_SPREAD':

            if msg.body['msg'] == 'QUERY_SPREAD':

                if self.mkt_closed: return
                # self.placeOrder()
                print(self.known_bids[self.symbol])
                print(self.known_asks[self.symbol])
                self.state = 'AWAITING_WAKEUP'


    def save_orderbook(self,  symbol, levels, freq):
        # super().requestDataSubscription(self, symbol, levels, freq)
        pass








