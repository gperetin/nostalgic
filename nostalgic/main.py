from dataclasses import dataclass
import datetime
from decimal import Decimal
from enum import Enum
import time
from typing import Callable, List, Dict, Optional

import pandas as pd

IndicatorFn = Callable[[pd.DataFrame], pd.DataFrame]

class HistoricalData:
    # container data type for historical data for one instrument
    # data is stored in a dataframe
    # we also keep the name of the market
    pass


class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP = 3


class Action:
    """Signal to the broker that something might need to happen.

    Actions are generated in the strategy by evaluating rules and are sent
    to the broker to signal that there could be something that needs to
    happen, such as fill or cancel an order.

    """

    # can be PlaceOrder, CancelOrder, ModifyOrder
    symbol: str


@dataclass
class PlaceOrder(Action):
    side: str
    quantity: int
    type_: OrderType
    price: Optional[Decimal] = None
    from_fill_price: Optional[Decimal] = None
    trailing_pct: float = None
    on_fill: List[Action] = None
    active_since: datetime.datetime = None

    def absolute_price(self, bars: pd.DataFrame) -> Decimal:
        """Returns the absolute price for this order.

        bars is a DataFrame with bars since this order is active.
        Since we use bars to compute HWM and LWM, they should not
        contain the bar that we're processing right now.
        That's not the current implementation, bars does contain the
        current bar as well. This is only a problem if the current
        bar both makes a new high (low) and moves in the opposite
        direction by trailing_pct percent. We'll take that risk for
        now, but proper implementation would not include the current
        bar.

        """

        if self.trailing_pct:
            # "long" means we're closing a short trade 
            if self.side == "long":
                lwm = bars.low.min()
                return lwm + lwm * self.trailing_pct
            else:
                hwm = bars.high.max()
                return hwm - hwm * self.trailing_pct

        return self.price


class Portfolio:
    pass


@dataclass
class Indicator:
    fn: IndicatorFn
    name: Optional[str] = None

    # If an indicator is global, it is available to all signals, regardless of
    # the instrument.
    is_global: bool = False

    # If symbols is set, the indicator is only defined for those symbols.
    symbols: List[str] = None


@dataclass
class Instrument:
    """Data object holding path-independent information for an instrument.

    Filters, indicators and signals are all precomputed and immutable.

    """

    symbol: str
    data: pd.DataFrame
    filters: List[pd.Series] = None
    indicators: Dict[str, pd.DataFrame] = None
    signals: Dict[str, pd.DataFrame] = None


class Rule:
    def __init__(self, instrument: Instrument):
        self._instrument = instrument
        self._current_date = None

    def evaluate(self) -> List[Action]:
        """Called by the strategy when the corresponding signal goes active.

        Has to be implemented by Rule subclass.

        """

        raise NotImplementedError

    def set_date(self, date: datetime.date):
        """Called by strategy to set the date for the current evaluation.

        This allows all the other get_* methods to work without passing in
        the date to each, provides nicer interface.

        """

        self._current_date = date

    def get_signal(self):
        """Return the signal that triggered the evaluation"""

        # TODO: currently, we only return the first signal - there's an assumption that only one signal
        # can trigger the rule. If we continue with that design, than we can accept signals and rules
        # as a dict in the strategy constructor
        sig_df = list(self._instrument.signals.values())[0]
        return sig_df.loc[self._current_date]

    def get_data(self):
        # Returns market data taking into account current date
        # implemented here
        pass

    def get_indicator(self, name: str) -> pd.DataFrame:
        """Return the requested indicator up to and including the current date."""

        return self._instrument.indicators[name][:self._current_date]


class Broker:
    # XXX: dummy implementation, only works for long side ATM
    def __init__(self, capital: Decimal, instruments: List[Instrument]):
        self._starting_capital = capital
        self._capital = capital
        self._positions = {} # str: tuple(size, price)
        self._queue: List[Action] = []

        self._data: Dict[str, pd.DataFrame] = {}
        for instr in instruments:
            self._data[instr.symbol] = instr.data

    def _buy(self, symbol, size, price):
        self._positions[symbol] = (size, price)

    def _sell(self, symbol, size, price):
        if symbol in self._positions:
            curr_pos, avg_price = self._positions[symbol]
            pnl = size * price - curr_pos * avg_price
            self._capital += pnl

            new_pos = curr_pos - size
            if new_pos == 0:
                del self._positions[symbol]
            elif new_pos > 0:
                self._positions[symbol] = (new_pos, avg_price)
            else:
                self._positions[symbol] = (new_pos, price)

            print("Pnl for {} is {}".format(symbol, pnl))
        else:
            # self._positions[symbol] = (size, price)
            pass

    def _next_bar(self, symbol: str, current_bar: datetime.datetime) -> datetime.datetime:
        """Returns the index of the next bar for which there's data for a given symbol."""

        # TODO: consider wrapping self._data into a type, in which case this
        # will go in that class

        instr_data = self._data[symbol]
        if current_bar not in instr_data.index:
            return None
        
        bars_after_current = instr_data[instr_data.index > current_bar]
        if len(bars_after_current):
            return bars_after_current.index[0].to_pydatetime()

        return None

    def add_action(self, action: Action):
        # TODO: come up with a better name for this
        self._queue.append(action)

    def open_positions(self):
        """Returns currently open positions."""

        return self._positions

    def run_loop(self, date: datetime.datetime):
        """Run the broker loop for a single bar that starts with `date`.

        Executes Actions if there are any.

        """

        # List of actions that will need to be executed on the next bar
        # We maintain this list because we pop from the queue of actions below.
        # Alternative is to traverse the list and remove if the action was executed.
        not_executed = []

        while self._queue:
            action = self._queue.pop(0)
            # TODO: we assume fill price is next day open, which is current day by the time
            # this code runs
            instr_data = self._data[action.symbol]
            if date not in instr_data.index:
                not_executed.append(action)
                continue

            fill_price = instr_data.loc[date].open

            # This big if/else block tries to see if an action can be executed (order can
            # be filled). This could be extracted onto the Order object (or Action)
            # Inputs: current bar, if it's a relative order, then bars since active
            # Returns: whether it was filled
            was_filled = False
            if action.type_ == OrderType.MARKET:
                if action.side == "long":
                    self._buy(action.symbol, action.quantity, fill_price)
                elif action.side == "sell":
                    self._sell(action.symbol, action.quantity, fill_price)
                was_filled = True
            elif action.type_ == OrderType.LIMIT:
                price_range = (instr_data.loc[date].low, instr_data.loc[date].high)

                if action.side == "long" and price_range[0] < action.price:
                    self._buy(action.symbol, action.quantity, action.price)
                    was_filled = True
                elif action.side == "sell" and price_range[1] > action.price:
                    self._sell(action.symbol, action.quantity, action.price)
                    was_filled = True
            elif action.type_ == OrderType.STOP:
                price_range = (instr_data.loc[date].low, instr_data.loc[date].high)

                # This is used for trailing stops
                # The slice access in Pandas in inclusive, so this includes `date`
                # bar as well
                # TODO: this currently sends the current bar as well, which is not
                # what we want
                bars_since_active = instr_data[action.active_since:date]

                # Price that has to be touched for this order to trigger
                # In case of an order with fixes price, that is action.price
                # For trailing orders, it's computed
                absolute_price = action.absolute_price(bars_since_active)
                if action.side == "long":
                    if price_range[1] > absolute_price:
                        self._buy(action.symbol, action.quantity, absolute_price)
                        was_filled = True
                else:
                    if price_range[0] < absolute_price:
                        self._sell(action.symbol, action.quantity, absolute_price)
                        was_filled = True
            else:
                raise ValueError("Invalid order type {}".format(action.type))

            # If the Action was executed, execute `on_fill` actions
            if was_filled:
                for fill_action in action.on_fill or []:
                    # TODO: Assigning a symbol here seems like a hack. I think that's because rule
                    # (where these were created) doesn't know about the symbol it is evaluating,
                    # seems like it should
                    fill_action.symbol = action.symbol

                    # If this new action has a price relative to the order we just executed
                    # compute the price here
                    if fill_action.from_fill_price:
                        fill_action.price = fill_price + fill_action.from_fill_price

                    # We also have to set the active_since here to be the next bar
                    fill_action.active_since = self._next_bar(action.symbol, date)

                    if not fill_action.active_since:
                        # This can happen if the current bar is the last bar in the data series
                        # In that case, we're not adding this action (Note: not sure if this is
                        # the best way to handle that)
                        continue

                    not_executed.append(fill_action)

                print("Executed action {} @ {}".format(action, fill_price))
            else:
                not_executed.append(action)

        self._queue = not_executed

class Strategy:
    def __init__(self,
            filters: List[Callable],
            indicators: List[Indicator],
            signals: Dict[str, Callable],
            rules: List[Rule],
            max_open_positions: int = 0
            ):

        self._filters = filters
        self._indicators = indicators
        self._signals = signals
        self._rules = rules

        self._instruments = None
        self._broker = None
        self._max_open_positions = max_open_positions

    def initialize(self, instruments: List[Instrument], broker: Broker):
        self._calculate_filters(instruments)
        global_indicators = self._calculate_indicators(instruments)
        self._calculate_signals(instruments, global_indicators)
        self._instruments = instruments
        self._broker = broker

    def _calculate_filters(self, instruments: List[Instrument]):
        for instr in instruments:
            instr.filters = []
            for filter_fn in self._filters:
                # Each filter fn returns a pd.Series of bools
                instr.filters.append(filter_fn(instr.data))

            # Let's also compute the combined output of all filters
            # If one filter is True (meaning, this security is filtered out),
            # the entire result is True
            # The combined filter will always be the last one
            filters_df = pd.concat(instr.filters, axis=1)
            instr.filters.append(filters_df.any(axis=1))

    def _calculate_indicators(self, instruments: List[Instrument]) -> Dict[str, pd.DataFrame]:
        """Calculate indicators for given instruments.

        Indicators are path-independent and are precomputed for all instruments
        before the strategy starts running.

        Returns a dict of computed global indicators.

        """

        global_indicators = {}

        for instr in instruments:
            instr.indicators = {}
            for ind in self._indicators:
                if ind.symbols and instr.symbol not in ind.symbols:
                    continue

                computed_ind = ind.fn(instr.data)
                instr.indicators[ind.name] = computed_ind

                if ind.is_global:
                    global_indicators[ind.name] = computed_ind

        return global_indicators

    def _calculate_signals(self, instruments: List[Instrument], global_indicators: Dict[str, pd.DataFrame] = None):
        """Calculate signals for given instruments.

        Signals are path-independent and are precomputed for all instruments
        before the stategy starts running.

        """
        for instr in instruments:
            instr.signals = {}
            for sig_name, fn in self._signals.items():
                instr.signals[sig_name] = fn(instr.indicators, global_indicators)

    def _position_for(self, symbol: str) -> Optional[int]:
        """Returns position for a symbol if it exists."""

        open_positions = self._broker.open_positions()
        return open_positions.get(symbol, None)

    def _num_open_positions(self) -> int:
        """Returns the number of open positions."""

        return len(self._broker.open_positions())

    def next(self, date: datetime.datetime):
        """Run the strategy for a given day.

        Checks all signals on all instruments, and if some triggered, runs the
        rules for those signals.
        Orders actions by the strength of the signal and sends them to broker.

        """

        self._broker.run_loop(date)

        actions = [] # List of tuples, (rank, action)
        for instr in self._instruments:
            # If we don't have price data for current date for instrument, skip
            if date not in instr.data.index:
                continue

            combined_filter = instr.filters[-1] if instr.filters else None

            for sig_name, sig_df in instr.signals.items():
                if date not in sig_df:
                    continue

                # If this instrument is filtered out for this date, skip rule evaluation
                if combined_filter is not None and combined_filter.loc[date]:
                    continue

                signal = sig_df.loc[date]

                for rule_class in self._rules:
                    rule = rule_class(instr)
                    rule.set_date(date) # TODO: if we create a rule for each evaluation, should we just pass in the date into the constructor too?
                    instr_actions = rule.evaluate()

                    for action in instr_actions:
                        # Rule doesn't have to know the symbol it is evaluating for, so we assign it here
                        action.symbol = instr.symbol
                        actions.append((signal, action))

        ranked_actions = sorted(actions, key=lambda elem: elem[0], reverse=True)
        for rank, action in ranked_actions:
            # This is a form of risk management - for thoughts on better way to do this, check
            # Risk management section in the notes
            if not self._position_for(action.symbol) and self._num_open_positions() >= self._max_open_positions:
                # We don't have a position and we're at or over the limit
                continue

            self._broker.add_action(action)
            print("[{}] [{}] Added action {} with rank {}".format(
                date, action.symbol, action, rank
            ))


class Backtest:
    # Not really sure what this is for yet, could be just processing the results and displaying

    def __init__(self, strategy: Strategy, instruments: List[Instrument], broker: Broker):
        self._strategy = strategy
        self._strategy.initialize(instruments, broker)
        self._broker = broker

    def run(self, date_range: pd.DatetimeIndex):
        for date in date_range:
            self._strategy.next(date)

        print("Ending cap: {}".format(self._broker._capital))
        print("Open positions")
        for position in self._broker._positions:
            print("Position: {}, {}".format(position, self._broker._positions[position][0]))
        print("Outstanding orders: {}")
        for order in self._broker._queue:
            print("Order {}".format(order))
