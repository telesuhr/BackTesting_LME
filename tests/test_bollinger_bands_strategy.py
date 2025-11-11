"""
ボリンジャーバンド戦略のテストコード

TDD原則：このテストを先に作成し、失敗を確認してから実装を進める
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBollingerBandsStrategy:
    """ボリンジャーバンド戦略のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成"""
        # 1週間分の1分足データ（7000行）
        dates = pd.date_range(
            start='2025-11-04 00:00:00',
            periods=7000,
            freq='1min'
        )

        # 平均回帰的な価格データを生成（ボリンジャーバンドに適した動き）
        np.random.seed(42)
        base_price = 10000

        # ランダムウォーク + 平均回帰成分
        returns = np.random.randn(7000) * 0.001  # 0.1%の標準偏差
        prices = base_price * np.exp(np.cumsum(returns))

        # 平均回帰させる（移動平均に戻る傾向）
        ma = pd.Series(prices).rolling(window=50).mean().fillna(prices[0])
        prices = prices * 0.7 + ma.values * 0.3

        close = prices
        high = close + np.abs(np.random.randn(7000) * 10)
        low = close - np.abs(np.random.randn(7000) * 10)
        open_price = close + np.random.randn(7000) * 5
        volume = np.random.randint(1, 100, 7000)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    def test_calculate_bollinger_bands(self, sample_data):
        """
        期待される動作：
        - ボリンジャーバンド（上限・中央・下限）を正しく計算できる
        - 中央線は移動平均
        - 上限・下限は中央線±2標準偏差
        """
        from src.strategy.bollinger_bands import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0
        )

        # ボリンジャーバンドを計算
        upper, middle, lower = strategy.calculate_bollinger_bands(sample_data)

        # 検証1: バンドが計算されていること
        assert upper is not None, "上限バンドが計算されていません"
        assert middle is not None, "中央線が計算されていません"
        assert lower is not None, "下限バンドが計算されていません"

        assert len(upper) == len(sample_data), "上限バンドの長さが不正です"
        assert len(middle) == len(sample_data), "中央線の長さが不正です"
        assert len(lower) == len(sample_data), "下限バンドの長さが不正です"

        # 検証2: バンドの順序が正しいこと（upper > middle > lower）
        valid_data = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_data] >= middle[valid_data]).all(), \
            "上限が中央線より下になっています"
        assert (middle[valid_data] >= lower[valid_data]).all(), \
            "中央線が下限より下になっています"

        # 検証3: 中央線が移動平均であること
        expected_ma = sample_data['close'].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            middle, expected_ma,
            check_names=False,
            rtol=1e-10
        )

        logger.info(f"✓ ボリンジャーバンド計算テスト成功")
        logger.info(f"  平均バンド幅: {(upper - lower).mean():.2f}")

    def test_generate_signals(self, sample_data):
        """
        期待される動作：
        - 価格が下限バンドにタッチしたらロングシグナル（1）
        - 価格が上限バンドにタッチしたらショートシグナル（-1）
        - それ以外はニュートラル（0）
        """
        from src.strategy.bollinger_bands import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0
        )

        # シグナルを生成
        signals = strategy.generate_signals(sample_data)

        # 検証1: シグナルが生成されていること
        assert signals is not None, "シグナルが生成されていません"
        assert len(signals) == len(sample_data), "シグナルの長さがデータと一致しません"

        # 検証2: シグナルが-1, 0, 1のいずれかであること
        unique_signals = signals.dropna().unique()
        assert set(unique_signals).issubset({-1, 0, 1}), \
            f"シグナルに想定外の値が含まれています: {unique_signals}"

        # 検証3: シグナルが生成されていること（全てが0ではない）
        assert (signals != 0).any(), "シグナルが全て0です"

        logger.info(f"✓ シグナル生成テスト成功")
        logger.info(f"  ロングシグナル: {(signals == 1).sum()}回")
        logger.info(f"  ショートシグナル: {(signals == -1).sum()}回")
        logger.info(f"  ニュートラル: {(signals == 0).sum()}回")

    def test_entry_exit_logic(self, sample_data):
        """
        期待される動作：
        - エントリーシグナルで正しくポジションを持つ
        - 利益確定・損切りで正しく決済
        - 同時に複数ポジションを持たない
        """
        from src.strategy.bollinger_bands import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0,
            take_profit_pct=0.01,  # 1%利益確定
            stop_loss_pct=0.01,     # 1%損切り
            max_positions=1
        )

        # バックテストを実行
        result = strategy.backtest(
            data=sample_data,
            initial_capital=1000000,
            risk_per_trade=0.02
        )

        # 検証1: バックテスト結果が返されること
        assert result is not None, "バックテスト結果が返されていません"
        assert 'trades' in result, "トレード情報が含まれていません"

        trades = result['trades']
        assert len(trades) > 0, "トレードが1件も実行されていません"

        # 検証2: 各トレードに必要な情報が含まれること
        required_fields = ['entry_time', 'exit_time', 'entry_price',
                          'exit_price', 'position', 'pnl']
        for trade in trades:
            for field in required_fields:
                assert field in trade, f"{field}が含まれていません"

        # 検証3: P&Lが計算されていること
        total_pnl = sum(trade['pnl'] for trade in trades)
        logger.info(f"✓ エントリー・エグジットロジックテスト成功")
        logger.info(f"  総トレード数: {len(trades)}回")
        logger.info(f"  総損益: {total_pnl:,.2f}円")

    def test_trading_costs(self, sample_data):
        """
        期待される動作：
        - 取引コストが正しく計算されること
        - ブローカー手数料: $0.5/片道
        - スプレッド: 0.01%
        """
        from src.strategy.bollinger_bands import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0,
            broker_commission_usd=0.5,
            spread_pct=0.0001
        )

        # 取引コストを計算
        entry_price = 10000.0
        exit_price = 10100.0
        position_size = 1.0

        costs = strategy.calculate_trading_costs(
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size
        )

        # 検証1: コストが正の値であること
        assert costs > 0, "取引コストが0以下です"

        # 検証2: コストの内訳が正しいこと
        # 手数料: $0.5 * 2 = $1.0 = 150円
        # スプレッド: (10000 * 0.0001 + 10100 * 0.0001) * 1 = 2.01
        # スプレッドUSD: 2.01 / 1 = 2.01
        # 総コストUSD: 1.0 + 2.01 = 3.01
        # 総コストJPY: 3.01 * 150 = 451.5
        expected_min = 400  # 許容範囲
        expected_max = 500
        assert expected_min <= costs <= expected_max, \
            f"取引コストが想定範囲外です: {costs}円 (期待: {expected_min}-{expected_max}円)"

        logger.info(f"✓ 取引コスト計算テスト成功")
        logger.info(f"  取引コスト: {costs:.2f}円")

    def test_mean_reversion_behavior(self, sample_data):
        """
        期待される動作：
        - ボリンジャーバンド戦略は平均回帰型
        - 下限タッチでロング → 価格が平均に戻ることを期待
        - 上限タッチでショート → 価格が平均に戻ることを期待
        """
        from src.strategy.bollinger_bands import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0
        )

        result = strategy.backtest(
            data=sample_data,
            initial_capital=1000000,
            risk_per_trade=0.02
        )

        # 検証1: パフォーマンス指標が計算されていること
        assert 'win_rate' in result, "勝率が計算されていません"
        assert 'total_return' in result, "総リターンが計算されていません"
        assert 'max_drawdown' in result, "最大ドローダウンが計算されていません"

        # 検証2: 勝率が計算されていること（値は問わない）
        assert 0 <= result['win_rate'] <= 1, \
            f"勝率が範囲外です: {result['win_rate']}"

        logger.info(f"✓ 平均回帰動作テスト成功")
        logger.info(f"  勝率: {result['win_rate']:.2%}")
        logger.info(f"  総リターン: {result['total_return']:.2%}")
        logger.info(f"  最大DD: {result['max_drawdown']:.2%}")


if __name__ == '__main__':
    # テストを実行
    pytest.main([__file__, '-v', '-s'])
