"""
LME Copper ボラティリティブレイクアウト戦略 バックテストスクリプト

使用方法:
    python run_backtest.py
"""
import logging
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import psycopg2

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy.volatility_breakout import VolatilityBreakoutStrategy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_db(ric_code='CMCU3', interval='1min'):
    """
    データベースからLME Copperデータを読み込む

    Args:
        ric_code: RICコード
        interval: データ間隔

    Returns:
        OHLCV DataFrameまたはNone
    """
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='lme_copper_db',
            user='postgres',
            password=''
        )

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM lme_copper_intraday_data
            WHERE ric_code = %s
              AND interval = %s
              AND open IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=(ric_code, interval))
        conn.close()

        if df.empty:
            logger.error(f"データが見つかりません: {ric_code} ({interval})")
            return None

        # timestampをインデックスに設定
        df.set_index('timestamp', inplace=True)

        # データ型を変換
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)

        logger.info(f"データ読み込み成功: {ric_code}")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def print_backtest_summary(result):
    """
    バックテスト結果のサマリーを表示

    Args:
        result: バックテスト結果辞書
    """
    print("\n" + "=" * 80)
    print("バックテスト結果サマリー")
    print("=" * 80)

    # 基本統計
    print(f"\n【基本統計】")
    print(f"  総トレード数:     {result['total_trades']:>6}回")
    print(f"  勝ちトレード:     {result['winning_trades']:>6}回")
    print(f"  負けトレード:     {result['losing_trades']:>6}回")
    print(f"  勝率:             {result['win_rate']:>6.2%}")

    # 損益
    print(f"\n【損益】")
    print(f"  初期資本:         {result['final_capital'] - result['total_pnl']:>12,.0f}円")
    print(f"  最終資本:         {result['final_capital']:>12,.0f}円")
    print(f"  総損益:           {result['total_pnl']:>12,.0f}円")
    print(f"  総リターン:       {result['total_return']:>12.2%}")

    # 手数料の内訳
    if 'total_trading_costs' in result:
        print(f"\n【取引コスト】")
        print(f"  総取引コスト:     {result['total_trading_costs']:>12,.0f}円")
        print(f"  取引回数:         {result['total_trades']:>12}回")
        print(f"  平均コスト/回:    {result['total_trading_costs'] / result['total_trades'] if result['total_trades'] > 0 else 0:>12,.0f}円")

    # トレード統計
    if result['winning_trades'] > 0:
        print(f"\n【トレード統計】")
        print(f"  平均利益:         {result['avg_win']:>12,.0f}円")
        if result['losing_trades'] > 0:
            print(f"  平均損失:         {result['avg_loss']:>12,.0f}円")
            print(f"  損益比率:         {abs(result['avg_win'] / result['avg_loss']):>12.2f}")

    # リスク指標
    print(f"\n【リスク指標】")
    print(f"  最大ドローダウン: {result['max_drawdown']:>12.2%}")
    print(f"  シャープレシオ:   {result['sharpe_ratio']:>12.2f}")

    print("\n" + "=" * 80)


def print_trade_details(trades, max_trades=10):
    """
    個別トレードの詳細を表示

    Args:
        trades: トレードリスト
        max_trades: 表示する最大トレード数
    """
    print(f"\n【直近{min(max_trades, len(trades))}件のトレード詳細】")
    print("-" * 150)
    print(f"{'#':<4} {'エントリー時刻':<20} {'エグジット時刻':<20} {'ポジション':<8} "
          f"{'エントリー価格':<12} {'エグジット価格':<12} {'総損益':<12} {'手数料':<10} {'純損益':<12} {'理由':<12}")
    print("-" * 150)

    for i, trade in enumerate(trades[-max_trades:], 1):
        position_str = 'ロング' if trade['position'] == 1 else 'ショート'

        # 総損益（手数料込み前）
        pnl_gross = trade.get('pnl_gross', trade['pnl'])
        pnl_gross_str = f"{pnl_gross:>,.0f}円"
        if pnl_gross > 0:
            pnl_gross_str = f"+ {pnl_gross_str}"
        else:
            pnl_gross_str = f"- {abs(pnl_gross):>,.0f}円"

        # 手数料
        costs = trade.get('trading_costs', 0)
        costs_str = f"{costs:>,.0f}円"

        # 純損益
        pnl_str = f"{trade['pnl']:>,.0f}円"
        if trade['pnl'] > 0:
            pnl_str = f"+ {pnl_str}"
        else:
            pnl_str = f"- {abs(trade['pnl']):>,.0f}円"

        print(f"{i:<4} {str(trade['entry_time']):<20} {str(trade['exit_time']):<20} "
              f"{position_str:<8} {trade['entry_price']:>10,.2f} {trade['exit_price']:>10,.2f} "
              f"{pnl_gross_str:<12} {costs_str:<10} {pnl_str:<12} {trade['exit_reason']:<12}")

    print("-" * 150)


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("LME Copper ボラティリティブレイクアウト戦略")
    logger.info("=" * 60)

    # データベースからデータを読み込む
    data = load_data_from_db(ric_code='CMCU3', interval='1min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # 戦略パラメータ
    strategy = VolatilityBreakoutStrategy(
        atr_period=14,           # ATR計算期間
        atr_multiplier=2.5,      # ブレイクアウト判定のATR倍率
        lookback_period=30,      # 価格変動を見る期間（分）
        stop_loss_atr=2.0,       # ストップロスのATR倍率
        take_profit_atr=4.0,     # 利益確定のATR倍率
        max_positions=1          # 最大同時ポジション数
    )

    logger.info("\n戦略パラメータ:")
    logger.info(f"  ATR期間:               {strategy.atr_period}")
    logger.info(f"  ATR倍率:               {strategy.atr_multiplier}")
    logger.info(f"  Lookback期間:          {strategy.lookback_period}分")
    logger.info(f"  ストップロス:          {strategy.stop_loss_atr} x ATR")
    logger.info(f"  利益確定:              {strategy.take_profit_atr} x ATR")
    logger.info(f"  最大ポジション数:      {strategy.max_positions}")

    # バックテスト実行
    result = strategy.backtest(
        data=data,
        initial_capital=1000000,  # 100万円
        risk_per_trade=0.02       # 1トレード当たり2%のリスク
    )

    # 結果を表示
    print_backtest_summary(result)

    if result['total_trades'] > 0:
        print_trade_details(result['trades'], max_trades=15)

    # データを可視化する場合は以下を有効化
    # plot_results(data, result)


def plot_results(data, result):
    """
    バックテスト結果を可視化（オプション）

    Args:
        data: 価格データ
        result: バックテスト結果
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 価格チャートとトレードポイントをプロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 価格チャート
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=1)

        # トレードポイントをマーク
        for trade in result['trades']:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            if trade['position'] == 1:  # ロング
                ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100, zorder=5)
                ax1.scatter(exit_time, exit_price, color='red', marker='v', s=100, zorder=5)
            else:  # ショート
                ax1.scatter(entry_time, entry_price, color='red', marker='v', s=100, zorder=5)
                ax1.scatter(exit_time, exit_price, color='green', marker='^', s=100, zorder=5)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD/MT)')
        ax1.set_title('LME Copper 3M - Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 資産曲線（簡易版）
        # 実装: result['equity_curve']がある場合
        # ax2.plot(equity_curve_times, equity_curve_values)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Capital (JPY)')
        ax2.set_title('Equity Curve')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        logger.info("\nチャートを保存しました: backtest_results.png")

    except ImportError:
        logger.warning("matplotlibがインストールされていないため、チャートを表示できません")


if __name__ == '__main__':
    main()
