"""
ボリンジャーバンド+RSI組み合わせ戦略（平均回帰型）

ボリンジャーバンドとRSIの両方が売買シグナルを出した時のみエントリー
高精度だがエントリー頻度は低い
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BollingerRSICombinedStrategy:
    """
    ボリンジャーバンド+RSI組み合わせ戦略

    戦略ロジック：
    1. ボリンジャーバンド2σタッチ AND RSI売買ゾーン（両方一致時のみエントリー）
    2. エグジット: エントリー価格から±2σ（ボリンジャーバンド戦略と同じ）
    3. 高精度エントリーを狙う
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        max_positions: int = 1,
        broker_commission_usd: float = 0.5,
        spread_pct: float = 0.0001,
        fixed_position_size: float = 100.0
    ):
        """
        Args:
            bb_period: ボリンジャーバンド計算期間
            bb_std: 標準偏差の倍率
            rsi_period: RSI計算期間
            rsi_oversold: 売られ過ぎ閾値
            rsi_overbought: 買われ過ぎ閾値
            max_positions: 最大同時ポジション数
            broker_commission_usd: ブローカー手数料（片道、USD）
            spread_pct: スプレッド（パーセンテージ）
            fixed_position_size: 固定ポジションサイズ（MT）
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
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

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        RSI（Relative Strength Index）を計算

        Args:
            data: OHLC価格データ

        Returns:
            RSI値のシリーズ
        """
        close = data['close']

        # 価格変動を計算
        delta = close.diff()

        # 上昇幅と下降幅を分離
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # 指数移動平均で平滑化
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        # RS = 平均上昇幅 / 平均下降幅
        rs = avg_gain / avg_loss

        # RSI = 100 - (100 / (1 + RS))
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        トレーディングシグナルを生成（AND条件）

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

        # RSIを計算
        rsi = self.calculate_rsi(data)

        signals = pd.Series(0, index=data.index)

        # ロングシグナル: ボリンジャー下限タッチ AND RSI < 30
        bb_long = (low <= lower) | (close <= lower)
        rsi_long = (rsi < self.rsi_oversold)
        signals[bb_long & rsi_long] = 1

        # ショートシグナル: ボリンジャー上限タッチ AND RSI > 70
        bb_short = (high >= upper) | (close >= upper)
        rsi_short = (rsi > self.rsi_overbought)
        signals[bb_short & rsi_short] = -1

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
        commission = self.broker_commission_usd * 2

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
        logger.info("ボリンジャーバンド+RSI組み合わせ戦略 バックテスト開始")
        logger.info("=" * 60)
        logger.info(f"初期資本: ${initial_capital:,.2f}")
        logger.info(f"ポジションサイズ: {self.fixed_position_size:,.1f} MT")
        logger.info(f"BB期間: {self.bb_period}, RSI期間: {self.rsi_period}")
        logger.info(f"データ期間: {data.index[0]} ～ {data.index[-1]}")
        logger.info(f"データ数: {len(data)}行")

        # ボリンジャーバンドを計算
        upper, middle, lower = self.calculate_bollinger_bands(data)

        # 標準偏差を計算（エグジット用）
        close = data['close']
        std = close.rolling(window=self.bb_period).std()

        # RSIを計算
        rsi = self.calculate_rsi(data)

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
            current_rsi = rsi.iloc[i]
            current_std = std.iloc[i]
            signal = signals.iloc[i]

            # NaN値のチェック
            if pd.isna(signal) or pd.isna(current_rsi) or pd.isna(current_std):
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
                        'entry_rsi': current_position['entry_rsi'],
                        'exit_rsi': current_rsi,
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
                # 固定ポジションサイズを使用
                position_size = self.fixed_position_size

                # エントリー（エントリー時の標準偏差とRSIを保存）
                current_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_std': current_std,
                    'entry_rsi': current_rsi,
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
        result['rsi_values'] = rsi
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower

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
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_trading_costs': 0.0
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
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)

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
