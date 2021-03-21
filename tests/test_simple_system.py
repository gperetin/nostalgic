"""Simple backtest.

This is a backtest for the first rule from the book Automated Stock Trading
Systems by Laurens Bensdorp.

This is work in progress, most of the code defined here will be moved to
other places, but for now we're also using this as an end-to-end test.

"""

import datetime
from decimal import Decimal
import os
from typing import Dict, List

import pandas as pd
import numpy as np

from nostalgic import Backtest, Broker, Indicator, IndicatorFn, Instrument, OrderType, PlaceOrder, Rule, Strategy


TEST_DATA_DIR = os.path.join("tests", "data")

def load_data(symbols) -> List[Instrument]:
    instrs = []
    for symbol in symbols:
        f_path = os.path.join(TEST_DATA_DIR, f"{symbol}.csv")
        df = pd.read_csv(f_path, parse_dates=["Date"], index_col=["Date"])
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"close": "unadjusted_close", "adj close": "close"})

        instr = Instrument(symbol, data=df)
        instrs.append(instr)
        
    return instrs


# Filters
# returns a Series of bool's
def min_price(data: pd.DataFrame) -> pd.Series:
    # Price has to be minimum $5
    return data.close < 5


def daily_volume(data: pd.DataFrame) -> pd.Series:
    volume = data.unadjusted_close * data.volume
    return volume.rolling(20).mean() < 50_000_000


# Indicators
def moving_avg(window) -> IndicatorFn:
    # a partial function that takes data argument(pd.DataFrame) and applies the MA calculation
    # column name is "ma_<window>"
    # use a partial function here to apply the window argument
    
    def calculate(data: pd.DataFrame) -> pd.DataFrame:
        res = data.close.rolling(window).mean()
        return pd.DataFrame({"ma_{}".format(window): res})
    
    return calculate


def mac_200(data: pd.DataFrame) -> pd.DataFrame:
    ma_200 = data.close.rolling(200).mean()
    return pd.DataFrame({"spy_mac": data.close > ma_200})


def pct_change(data: pd.DataFrame) -> pd.DataFrame:
    rate_of_change = data.close.pct_change(200)
    return pd.DataFrame({"rate_of_change": rate_of_change})


def atr_20(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "high": data.high,
        "yday_close": data.shift(1).close,
        "low": data.low
    })
    
    df["true_range"] = df[['high', 'yday_close']].max(axis=1) - df[['low', 'yday_close']].min(axis=1)
    df["avg_true_range"] = df["true_range"].rolling(20).mean()
    return pd.DataFrame({"atr": df["avg_true_range"]})


# Signals
def ma_crossover(
        indicators: Dict[str, pd.DataFrame],
        global_indicators: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
    
    # If 50 MA is over 200 MA, and SPY close is above 200 day MA, then return a buy signal
    # The value of the signal is the 200 day percent change, we call it rank here
    
    df = pd.DataFrame({
        "ma_cross": indicators["ma_50"].ma_50 - indicators["ma_200"].ma_200,
        "rate_of_change": indicators["pct_change"].rate_of_change,
        "spy_cross_200ma": global_indicators["spy_mac_200"].spy_mac,
    })
    
    df["ma_cross_change"] = df.ma_cross.apply(lambda x: 0 if x < 0 else 1).diff()
    return df[df["ma_cross_change"] > 0 & df["spy_cross_200ma"]].rate_of_change


class EntryRule(Rule):
    def evaluate(self):
        # Simple long-only rule, for every positive signal, buy 10 shares
        
        # TODO: If multiple signals can trigger the same rule, this needs to be changed
        signal = self.get_signal() # Signal that triggered the rule
        
        actions = []
        if signal > 0:
            current_atr = self.get_indicator("atr_20").iloc[-1].atr
            stop_loss_price_offset = -1 * current_atr * 5
            
            stop_loss_order = PlaceOrder("sell", 10, OrderType.STOP, from_fill_price=stop_loss_price_offset)
            profit_protection_order = PlaceOrder("sell", 10, OrderType.STOP, trailing_pct=0.25)
            entry_order = PlaceOrder("long", 10, OrderType.MARKET, on_fill=[stop_loss_order, profit_protection_order])
            actions.append(entry_order)
        
        return actions


def test_simple_system():
    symbols = ["GOOG", "LULU", "TWTR", "SPY"]
    instrs = load_data(symbols)

    strat = Strategy(
        filters=[min_price, daily_volume],
        indicators=[
            Indicator(moving_avg(window=50), "ma_50"),
            Indicator(moving_avg(window=200), "ma_200"),
            Indicator(pct_change, "pct_change"),
            Indicator(atr_20, "atr_20"),
            Indicator(mac_200, "spy_mac_200", is_global=True, symbols=["SPY"])
        ],
        signals={"mac": ma_crossover},
        rules=[EntryRule],
        max_open_positions = 10
    )

    date_range = pd.date_range(start="2020-04-01", end="2021-03-01")
    broker = Broker(100000, instrs)
    bt = Backtest(strat, instrs, broker)
    bt.run(date_range)
