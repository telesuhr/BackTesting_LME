"""
ボラティリティブレイクアウト戦略（ATRベース）

24時間取引のLME Copperに適したトレンドフォロー戦略
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy:
    """
    ボラティリティブレイクアウト戦略

    戦略ロジック：
    1. ATR（Average True Range）でボラティリティを測定
    2. 価格変動がATR × multiplierを超えた場合にエントリー
    3. ATRベースのストップロス・利益確定で決済
    4. トレンドフォロー型の戦略
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        lookback_period: int = 20,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
        max_positions: int = 1,
        broker_commission_usd: float = 0.5,  # ブローカー手数料（片道、USD）
        spread_pct: float = 0.0001  # スプレッド（0.01%）
    ):
        """
        Args:
            atr_period: ATR計算期間（デフォルト14）
            atr_multiplier: ブレイクアウト判定のATR倍率（デフォルト2.0）
            lookback_period: 価格変動を見る期間（デフォルト20）
            stop_loss_atr: ストップロスのATR倍率（デフォルト2.0）
            take_profit_atr: 利益確定のATR倍率（デフォルト3.0）
            max_positions: 最大同時ポジション数（デフォルト1）
            broker_commission_usd: ブローカー手数料（片道、USD）
            spread_pct: スプレッド（パーセンテージ）
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.lookback_period = lookback_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_positions = max_positions
        self.broker_commission_usd = broker_commission_usd
        self.spread_pct = spread_pct

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        ATR（Average True Range）を計算

        Args:
            data: OHLC価格データ

        Returns:
            ATR値のSeries
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Rangeを計算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATRは指数移動平均で計算
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        return atr

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        トレーディングシグナルを生成

        Args:
            data: OHLC価格データ

        Returns:
            シグナル（1: ロング, -1: ショート, 0: ニュートラル）
        """
        close = data['close']
        atr = self.calculate_atr(data)

        # 価格変動率を計算
        price_change = close - close.shift(self.lookback_period)

        # ATRを基準にブレイクアウトを判定
        breakout_threshold = atr * self.atr_multiplier

        signals = pd.Series(0, index=data.index)

        # ロングシグナル: 価格がATR × multiplier以上上昇
        signals[price_change > breakout_threshold] = 1

        # ショートシグナル: 価格がATR × multiplier以上下落
        signals[price_change < -breakout_threshold] = -1

        return signals

    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        current_price: float,
        atr: float
    ) -> float:
        """
        ATRベースでポジションサイズを計算

        Args:
            capital: 利用可能資本
            risk_per_trade: 1トレード当たりのリスク比率（例: 0.02 = 2%）
            current_price: 現在価格
            atr: 現在のATR値

        Returns:
            ポジションサイズ（ロット数）
        """
        # リスク許容額
        risk_amount = capital * risk_per_trade

        # ATRベースのストップロス幅（価格単位）
        stop_loss_distance = atr * self.stop_loss_atr

        # ポジションサイズ = リスク許容額 / ストップロス幅
        position_size = risk_amount / stop_loss_distance

        # 最小単位は1ロット
        return max(1.0, position_size)

    def calculate_trading_costs(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float
    ) -> float:
        """
        取引コストを計算（手数料 + スプレッド）

        Args:
            entry_price: エントリー価格
            exit_price: エグジット価格
            position_size: ポジションサイズ

        Returns:
            総取引コスト（円）
        """
        # ブローカー手数料（往復）
        # 注：LME Copperは1ロット=1トンではなく、価格がUSD/MTで表示されている
        # position_sizeは実際の取引量（資金ベース）として扱う
        commission = self.broker_commission_usd * 2  # 往復

        # スプレッドコスト（往復）
        spread_cost_entry = entry_price * self.spread_pct
        spread_cost_exit = exit_price * self.spread_pct
        spread_total = (spread_cost_entry + spread_cost_exit) * position_size

        # 総コスト（USD）
        total_cost_usd = commission + spread_total

        # USDからJPYに変換（簡易的に150円/USDを使用）
        # 実際の運用では為替レートも考慮が必要
        usd_jpy_rate = 150.0
        total_cost_jpy = total_cost_usd * usd_jpy_rate

        return total_cost_jpy

    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 1000000,
        risk_per_trade: float = 0.02
    ) -> Dict:
        """
        バックテストを実行

        Args:
            data: OHLC価格データ
            initial_capital: 初期資本
            risk_per_trade: 1トレード当たりのリスク比率

        Returns:
            バックテスト結果の辞書
        """
        logger.info("=" * 60)
        logger.info("ボラティリティブレイクアウト戦略 バックテスト開始")
        logger.info("=" * 60)
        logger.info(f"初期資本: {initial_capital:,.0f}円")
        logger.info(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        logger.info(f"データ数: {len(data)}行")

        # シグナルとATRを計算
        signals = self.generate_signals(data)
        atr = self.calculate_atr(data)

        # トレード記録
        trades = []
        current_position = None
        capital = initial_capital
        equity_curve = [initial_capital]

        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_atr = atr.iloc[i]
            signal = signals.iloc[i]

            # NaN値のチェック
            if pd.isna(current_atr) or pd.isna(signal):
                equity_curve.append(equity_curve[-1])
                continue

            # ポジションを持っている場合
            if current_position is not None:
                position_type = current_position['position']
                entry_price = current_position['entry_price']
                stop_loss = current_position['stop_loss']
                take_profit = current_position['take_profit']

                # エグジット判定
                should_exit = False
                exit_reason = None

                if position_type == 1:  # ロングポジション
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price >= take_profit:
                        should_exit = True
                        exit_reason = 'take_profit'
                elif position_type == -1:  # ショートポジション
                    if current_price >= stop_loss:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price <= take_profit:
                        should_exit = True
                        exit_reason = 'take_profit'

                if should_exit:
                    # 損益計算（手数料・スプレッド控除前）
                    if position_type == 1:
                        pnl_gross = (current_price - entry_price) * current_position['size']
                    else:
                        pnl_gross = (entry_price - current_price) * current_position['size']

                    # 取引コストを計算
                    trading_costs = self.calculate_trading_costs(
                        entry_price=entry_price,
                        exit_price=current_price,
                        position_size=current_position['size']
                    )

                    # 純損益
                    pnl = pnl_gross - trading_costs
                    capital += pnl

                    # トレード記録
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position_type,
                        'size': current_position['size'],
                        'pnl_gross': pnl_gross,
                        'trading_costs': trading_costs,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)

                    current_position = None

            # 新規エントリー判定
            if current_position is None and signal != 0:
                # ポジションサイズを計算
                position_size = self.calculate_position_size(
                    capital=capital,
                    risk_per_trade=risk_per_trade,
                    current_price=current_price,
                    atr=current_atr
                )

                # エントリー
                if signal == 1:  # ロングエントリー
                    stop_loss = current_price - (current_atr * self.stop_loss_atr)
                    take_profit = current_price + (current_atr * self.take_profit_atr)
                else:  # ショートエントリー
                    stop_loss = current_price + (current_atr * self.stop_loss_atr)
                    take_profit = current_price - (current_atr * self.take_profit_atr)

                current_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'position': signal,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }

            equity_curve.append(capital)

        # パフォーマンス指標を計算
        result = self._calculate_performance_metrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital
        )

        result['trades'] = trades

        logger.info("=" * 60)
        logger.info("バックテスト結果")
        logger.info("=" * 60)
        logger.info(f"総トレード数: {result['total_trades']}回")
        logger.info(f"勝率: {result['win_rate']:.2%}")
        logger.info(f"総損益: {result['total_pnl']:,.2f}円")
        logger.info(f"最終資本: {result['final_capital']:,.2f}円")
        logger.info(f"リターン: {result['total_return']:.2%}")
        logger.info(f"最大ドローダウン: {result['max_drawdown']:.2%}")
        logger.info(f"シャープレシオ: {result['sharpe_ratio']:.2f}")
        logger.info("=" * 60)

        return result

    def _calculate_performance_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        initial_capital: float
    ) -> Dict:
        """
        パフォーマンス指標を計算

        Args:
            trades: トレード記録のリスト
            equity_curve: 資産曲線
            initial_capital: 初期資本

        Returns:
            パフォーマンス指標の辞書
        """
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_capital': initial_capital,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trades': []
            }

        # 基本統計
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_pnl = sum(t['pnl'] for t in trades)
        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital

        # 最大ドローダウンを計算
        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # シャープレシオを計算
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)  # 年率換算

        # 総取引コストを計算
        total_trading_costs = sum(t.get('trading_costs', 0) for t in trades)

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'total_trading_costs': total_trading_costs
        }
