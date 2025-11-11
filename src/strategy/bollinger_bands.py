"""
ボリンジャーバンド戦略（平均回帰型）

24時間取引のLME Copperに適した平均回帰戦略
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BollingerBandsStrategy:
    """
    ボリンジャーバンド戦略

    戦略ロジック：
    1. 移動平均と標準偏差でボリンジャーバンドを計算
    2. 価格が下限バンドにタッチしたら買い（平均回帰を期待）
    3. 価格が上限バンドにタッチしたら売り（平均回帰を期待）
    4. 同時に利益確定・損切りオーダーを設定
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        take_profit_pct: float = 0.015,  # 1.5%利益確定
        stop_loss_pct: float = 0.015,    # 1.5%損切り
        max_positions: int = 1,
        broker_commission_usd: float = 0.5,  # ブローカー手数料（片道、USD）
        spread_pct: float = 0.0001,  # スプレッド（0.01%）
        fixed_position_size: float = 2500.0  # 固定ポジションサイズ（MT）100ロット = 2500MT
    ):
        """
        Args:
            bb_period: ボリンジャーバンド計算期間（デフォルト20）
            bb_std: 標準偏差の倍率（デフォルト2.0）
            take_profit_pct: 利益確定の価格変動率（デフォルト1.5%）
            stop_loss_pct: 損切りの価格変動率（デフォルト1.5%）
            max_positions: 最大同時ポジション数（デフォルト1）
            broker_commission_usd: ブローカー手数料（片道、USD）
            spread_pct: スプレッド（パーセンテージ）
            fixed_position_size: 固定ポジションサイズ（MT）
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        self.broker_commission_usd = broker_commission_usd
        self.spread_pct = spread_pct
        self.fixed_position_size = fixed_position_size

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ボリンジャーバンドを計算

        Args:
            data: OHLC価格データ

        Returns:
            (上限バンド, 中央線, 下限バンド)のタプル
        """
        close = data['close']

        # 中央線（移動平均）
        middle = close.rolling(window=self.bb_period).mean()

        # 標準偏差
        std = close.rolling(window=self.bb_period).std()

        # 上限・下限バンド
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)

        return upper, middle, lower

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        トレーディングシグナルを生成

        Args:
            data: OHLC価格データ

        Returns:
            シグナル（1: ロング, -1: ショート, 0: ニュートラル）
        """
        close = data['close']
        low = data['low']
        high = data['high']

        # ボリンジャーバンドを計算
        upper, middle, lower = self.calculate_bollinger_bands(data)

        signals = pd.Series(0, index=data.index)

        # ロングシグナル: 価格が下限バンドにタッチ（平均回帰を期待）
        # 安値が下限バンドに触れたか、終値が下限バンド以下
        signals[(low <= lower) | (close <= lower)] = 1

        # ショートシグナル: 価格が上限バンドにタッチ（平均回帰を期待）
        # 高値が上限バンドに触れたか、終値が上限バンド以上
        signals[(high >= upper) | (close >= upper)] = -1

        return signals

    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        current_price: float
    ) -> float:
        """
        ポジションサイズを計算（MT単位）

        Args:
            capital: 利用可能資本（円）
            risk_per_trade: 1トレード当たりのリスク比率（例: 0.02 = 2%）
            current_price: 現在価格（USD/MT）

        Returns:
            ポジションサイズ（メトリックトン数）
        """
        # リスク許容額（円）
        risk_amount = capital * risk_per_trade

        # 損切り幅（USD/MT）
        stop_loss_distance = current_price * self.stop_loss_pct

        # USD/JPY為替レート
        usd_jpy_rate = 150.0

        # 損切り幅（円/MT）
        stop_loss_distance_jpy = stop_loss_distance * usd_jpy_rate

        # ポジションサイズ = リスク許容額（円） / 損切り幅（円/MT）
        position_size = risk_amount / stop_loss_distance_jpy

        # 最小単位は1MT、最大50MT（現実的な範囲）
        return max(1.0, min(position_size, 50.0))

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
        logger.info("ボリンジャーバンド戦略 バックテスト開始")
        logger.info("=" * 60)
        logger.info(f"初期資本: ${initial_capital:,.2f}")
        logger.info(f"ポジションサイズ: {self.fixed_position_size:,.1f} MT")
        logger.info(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        logger.info(f"データ数: {len(data)}行")

        # ボリンジャーバンドを計算（2σと3σ）
        upper_2sigma, middle, lower_2sigma = self.calculate_bollinger_bands(data)

        # 3σバンドを計算
        close = data['close']
        std = close.rolling(window=self.bb_period).std()
        upper_3sigma = middle + (std * 3.0)
        lower_3sigma = middle - (std * 3.0)

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

            # 現在のバンド値と標準偏差
            current_middle = middle.iloc[i]
            current_std = std.iloc[i]
            current_upper_3sigma = upper_3sigma.iloc[i]
            current_lower_3sigma = lower_3sigma.iloc[i]

            # NaN値のチェック
            if pd.isna(signal) or pd.isna(current_middle) or pd.isna(current_std):
                equity_curve.append(equity_curve[-1])
                continue

            # ポジションを持っている場合
            if current_position is not None:
                position_type = current_position['position']
                entry_price = current_position['entry_price']
                entry_std = current_position['entry_std']

                # エグジット判定（エントリー位置から2σ分の変動で判定）
                should_exit = False
                exit_reason = None

                # 2σ分の価格変動
                two_sigma_move = entry_std * 2.0

                if position_type == 1:  # ロングポジション
                    # 利食い：エントリー価格から2σ上昇
                    if current_price >= entry_price + two_sigma_move:
                        should_exit = True
                        exit_reason = 'take_profit'
                    # 損切り：エントリー価格から2σ下落
                    elif current_price <= entry_price - two_sigma_move:
                        should_exit = True
                        exit_reason = 'stop_loss'

                elif position_type == -1:  # ショートポジション
                    # 利食い：エントリー価格から2σ下落
                    if current_price <= entry_price - two_sigma_move:
                        should_exit = True
                        exit_reason = 'take_profit'
                    # 損切り：エントリー価格から2σ上昇
                    elif current_price >= entry_price + two_sigma_move:
                        should_exit = True
                        exit_reason = 'stop_loss'

                if should_exit:
                    # 損益計算（手数料・スプレッド控除前、USD）
                    # (価格差 USD/MT) * (MT数) = USD
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
            if current_position is None and signal != 0:
                # 固定ポジションサイズを使用（100ロット = 2500MT）
                position_size = self.fixed_position_size

                # エントリー（エントリー時の標準偏差を保存）
                current_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_std': current_std,
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
