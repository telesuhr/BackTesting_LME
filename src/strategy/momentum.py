"""
モメンタム戦略（移動平均クロスオーバー）

24時間取引のLME Copperに適したトレンドフォロー戦略
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """
    移動平均クロスオーバー戦略

    戦略ロジック：
    1. 短期移動平均と長期移動平均を計算
    2. ゴールデンクロス（短期が長期を上抜け）で買い
    3. デッドクロス（短期が長期を下抜け）で売り
    4. 逆クロスが発生したらポジションクローズ
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        take_profit_pct: float = 0.02,  # 2%利益確定
        stop_loss_pct: float = 0.01,    # 1%損切り
        max_positions: int = 1,
        broker_commission_usd: float = 0.5,  # ブローカー手数料（片道、USD）
        spread_pct: float = 0.0001,  # スプレッド（0.01%）
        fixed_position_size: float = 100.0  # 固定ポジションサイズ（MT）
    ):
        """
        Args:
            fast_period: 短期移動平均の期間（デフォルト5）
            slow_period: 長期移動平均の期間（デフォルト20）
            take_profit_pct: 利益確定の価格変動率（デフォルト2%）
            stop_loss_pct: 損切りの価格変動率（デフォルト1%）
            max_positions: 最大同時ポジション数（デフォルト1）
            broker_commission_usd: ブローカー手数料（片道、USD）
            spread_pct: スプレッド（パーセンテージ）
            fixed_position_size: 固定ポジションサイズ（MT）
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        self.broker_commission_usd = broker_commission_usd
        self.spread_pct = spread_pct
        self.fixed_position_size = fixed_position_size

    def calculate_moving_averages(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        移動平均を計算

        Args:
            data: OHLC価格データ

        Returns:
            (短期移動平均, 長期移動平均)のタプル
        """
        close = data['close']

        # 短期移動平均
        fast_ma = close.rolling(window=self.fast_period).mean()

        # 長期移動平均
        slow_ma = close.rolling(window=self.slow_period).mean()

        return fast_ma, slow_ma

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        トレーディングシグナルを生成

        Args:
            data: OHLC価格データ

        Returns:
            シグナル（1: ロング, -1: ショート, 0: ニュートラル）
        """
        fast_ma, slow_ma = self.calculate_moving_averages(data)

        signals = pd.Series(0, index=data.index)

        # クロスオーバーを検出
        # ゴールデンクロス: 短期が長期を上抜け（前の足で短期<長期、現在の足で短期>長期）
        golden_cross = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        signals[golden_cross] = 1

        # デッドクロス: 短期が長期を下抜け（前の足で短期>長期、現在の足で短期<長期）
        dead_cross = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
        signals[dead_cross] = -1

        return signals

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
            総取引コスト（USD）
        """
        # ブローカー手数料（往復）
        commission = self.broker_commission_usd * 2  # 往復

        # スプレッドコスト（往復）
        spread_cost_entry = entry_price * self.spread_pct
        spread_cost_exit = exit_price * self.spread_pct
        spread_total = (spread_cost_entry + spread_cost_exit) * position_size

        # 総コスト（USD）
        total_cost_usd = commission + spread_total

        return total_cost_usd

    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
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
        logger.info("モメンタム戦略（移動平均クロスオーバー） バックテスト開始")
        logger.info("=" * 60)
        logger.info(f"初期資本: ${initial_capital:,.2f}")
        logger.info(f"ポジションサイズ: {self.fixed_position_size:,.1f} MT")
        logger.info(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        logger.info(f"データ数: {len(data)}行")

        # 移動平均を計算
        fast_ma, slow_ma = self.calculate_moving_averages(data)

        # シグナルを計算
        signals = self.generate_signals(data)

        # トレード記録
        trades = []
        current_position = None
        capital = initial_capital
        equity_curve = [initial_capital]

        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]

            # 移動平均値
            current_fast_ma = fast_ma.iloc[i]
            current_slow_ma = slow_ma.iloc[i]

            # NaN値のチェック
            if pd.isna(signal) or pd.isna(current_fast_ma) or pd.isna(current_slow_ma):
                equity_curve.append(equity_curve[-1])
                continue

            # ポジションを持っている場合
            if current_position is not None:
                position_type = current_position['position']
                entry_price = current_position['entry_price']

                # エグジット判定
                should_exit = False
                exit_reason = None

                # 利食い・損切り判定
                if position_type == 1:  # ロングポジション
                    # 利食い
                    if current_price >= entry_price * (1 + self.take_profit_pct):
                        should_exit = True
                        exit_reason = 'take_profit'
                    # 損切り
                    elif current_price <= entry_price * (1 - self.stop_loss_pct):
                        should_exit = True
                        exit_reason = 'stop_loss'
                    # デッドクロス（逆クロス）でエグジット
                    elif signal == -1:
                        should_exit = True
                        exit_reason = 'reverse_cross'

                elif position_type == -1:  # ショートポジション
                    # 利食い
                    if current_price <= entry_price * (1 - self.take_profit_pct):
                        should_exit = True
                        exit_reason = 'take_profit'
                    # 損切り
                    elif current_price >= entry_price * (1 + self.stop_loss_pct):
                        should_exit = True
                        exit_reason = 'stop_loss'
                    # ゴールデンクロス（逆クロス）でエグジット
                    elif signal == 1:
                        should_exit = True
                        exit_reason = 'reverse_cross'

                if should_exit:
                    # 損益計算（手数料・スプレッド控除前、USD）
                    if position_type == 1:
                        pnl_gross = (current_price - entry_price) * current_position['size']
                    else:
                        pnl_gross = (entry_price - current_price) * current_position['size']

                    # 取引コストを計算（USD）
                    trading_costs = self.calculate_trading_costs(
                        entry_price=entry_price,
                        exit_price=current_price,
                        position_size=current_position['size']
                    )

                    # 純損益（USD）
                    pnl = pnl_gross - trading_costs
                    capital += pnl

                    # 資本がマイナスになった場合の警告
                    if capital < 0:
                        logger.warning(f"警告: 資本がマイナスになりました (${capital:,.2f}) at {current_time}")

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
            # ポジションがなく、かつシグナルが発生した場合
            if current_position is None and signal != 0:
                # 固定ポジションサイズを使用
                position_size = self.fixed_position_size

                # エントリー
                current_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'position': signal,
                    'size': position_size
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
        logger.info(f"総損益: ${result['total_pnl']:,.2f}")
        logger.info(f"最終資本: ${result['final_capital']:,.2f}")
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
