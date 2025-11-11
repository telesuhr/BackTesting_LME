"""
Trading strategy modules
"""
from .volatility_breakout import VolatilityBreakoutStrategy
from .bollinger_bands import BollingerBandsStrategy

__all__ = ['VolatilityBreakoutStrategy', 'BollingerBandsStrategy']
