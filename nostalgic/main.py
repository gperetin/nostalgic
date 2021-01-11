from collections.abc import Callable
from dataclasses import dataclass
import datetime
from decimal import Decimal
from typing import List, Dict

import pandas as pd


class HistoricalData:
    # container data type for historical data for one instrument
    # data is stored in a dataframe
    # we also keep the name of the market
    pass


class Action:
    # can be PlaceOrder, CancelOrder, ModifyOrder
    pass


@dataclass
class PlaceOrder(Action):
    side: str
    quantity: int


class Portfolio:
    pass


@dataclass
class Instrument:
    # Data object holding information about an instrument
    # After filters, indicators and signals are computed, this should be immutable
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

    def get_indicators(self):
        # Returns indicators for current date
        # implemented here
        pass


class Broker:
    # XXX: dummy implementation, only works for long side ATM
    def __init__(self, capital: Decimal):
        self._starting_capital = capital
        self._capital = capital
        self._positions = {} # str: tuple(size, price)

    def buy(self, symbol, size, price):
        self._positions[symbol] = (size, price)

    def sell(self, symbol, size, price):
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


class Strategy:
    def __init__(self,
            filters: List[Callable],
            indicators: Dict[str, Callable],
            signals: Dict[str, Callable],
            rules: List[Rule]
            ):

        self._filters = filters
        self._indicators = indicators
        self._signals = signals
        self._rules = rules

        self._instruments = None
        self._broker = None

    def initialize(self, instruments: List[Instrument], broker: Broker):
        self._calculate_filters(instruments)
        self._calculate_indicators(instruments)
        self._calculate_signals(instruments)
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

    def _calculate_indicators(self, instruments: List[Instrument]):
        """Calculate indicators for given instruments.

        Indicators are path-independent and are precomputed for all instruments
        before the strategy starts running.

        """

        for instr in instruments:
            instr.indicators = {}
            for ind in self._indicators:
                fn = self._indicators[ind]
                instr.indicators[ind] = fn(instr.data)

    def _calculate_signals(self, instruments: List[Instrument]):
        """Calculate signals for given instruments.

        Signals are path-independent and are precomputed for all instruments
        before the stategy starts running.

        """
        for instr in instruments:
            instr.signals = {}
            for sig_name, fn in self._signals.items():
                instr.signals[sig_name] = fn(instr.indicators)

    def next(self, date):
        """Run the strategy for a given day.

        Checks all signals on all instruments, and if some triggered, runs the
        rules for those signals.

        """

        for instr in self._instruments:
            combined_filter = instr.filters[-1] if instr.filters else None

            for sig_name, sig_df in instr.signals.items():
                if date not in sig_df:
                    continue

                # If this instrument is filtered out for this date, skip rule evaluation
                if combined_filter is not None and combined_filter.loc[date]:
                    print("Skipping {} on {} because it's filtered out.".format(instr.symbol, date))
                    continue

                # Note: date here needs to be datetime object, or str like "2020-12-31"
                signal = sig_df.loc[date]

                for rule_class in self._rules:
                    rule = rule_class(instr)
                    rule.set_date(date) # TODO: if we create a rule for each evaluation, should we just pass in the date into the constructor too?
                    actions = rule.evaluate()
                    for action in actions:
                        fill_price = instr.data.loc[date].close
                        print("[{}] [{}] Would take action {} @ {}".format(
                            date, instr.symbol, action, fill_price
                        ))
                        if action.side == "long":
                            self._broker.buy(instr.symbol, action.quantity, fill_price)
                        elif action.side == "sell":
                            self._broker.sell(instr.symbol, action.quantity, fill_price)


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
