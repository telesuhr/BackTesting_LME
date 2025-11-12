"""
LME Copper 1日分バックテストスクリプト

使用方法:
    python run_backtest_single_day.py
"""
import logging
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import psycopg2

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy.bollinger_bands import BollingerBandsStrategy
from src.strategy.volatility_breakout import VolatilityBreakoutStrategy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_single_day_data(target_date='2025-11-10', ric_code='CMCU3', interval='1min'):
    """
    データベースから特定日のLME Copperデータを読み込む

    Args:
        target_date: 対象日（YYYY-MM-DD）
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

        # 対象日の00:00:00～23:59:59のデータを取得
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM lme_copper_intraday_data
            WHERE ric_code = %s
              AND interval = %s
              AND timestamp >= %s::timestamp
              AND timestamp < (%s::timestamp + INTERVAL '1 day')
              AND open IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=(ric_code, interval, target_date, target_date))
        conn.close()

        if df.empty:
            logger.error(f"データが見つかりません: {target_date}")
            return None

        # timestampをインデックスに設定
        df.set_index('timestamp', inplace=True)

        # データ型を変換
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)

        logger.info(f"データ読み込み成功: {target_date}")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def print_backtest_summary(strategy_name, result):
    """
    バックテスト結果のサマリーを表示

    Args:
        strategy_name: 戦略名
        result: バックテスト結果辞書
    """
    print("\n" + "=" * 80)
    print(f"{strategy_name} - バックテスト結果")
    print("=" * 80)

    # 基本統計
    print(f"\n【基本統計】")
    print(f"  総トレード数:     {result['total_trades']:>6}回")

    if result['total_trades'] == 0:
        print("  ※ トレードが発生しませんでした")
        print("\n" + "=" * 80)
        return

    print(f"  勝ちトレード:     {result.get('winning_trades', 0):>6}回")
    print(f"  負けトレード:     {result.get('losing_trades', 0):>6}回")
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
        if result['total_trades'] > 0:
            print(f"  平均コスト/回:    {result['total_trading_costs'] / result['total_trades']:>12,.0f}円")

            # 手数料を除いた総損益
            pnl_before_costs = result['total_pnl'] + result['total_trading_costs']
            print(f"  手数料控除前損益:   {pnl_before_costs:>12,.0f}円")

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


def print_trade_details(trades):
    """
    全トレードの詳細を表示

    Args:
        trades: トレードリスト
    """
    if len(trades) == 0:
        print("\n【トレード詳細】")
        print("  トレードが発生しませんでした")
        return

    print(f"\n【全トレード詳細 ({len(trades)}件)】")
    print("-" * 150)
    print(f"{'#':<4} {'エントリー時刻':<20} {'エグジット時刻':<20} {'ポジション':<8} "
          f"{'エントリー価格':<12} {'エグジット価格':<12} {'総損益':<12} {'手数料':<10} {'純損益':<12} {'理由':<12}")
    print("-" * 150)

    for i, trade in enumerate(trades, 1):
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
    target_date = '2025-11-10'

    logger.info("=" * 60)
    logger.info(f"LME Copper 1日分バックテスト: {target_date}")
    logger.info("=" * 60)

    # データベースからデータを読み込む
    data = load_single_day_data(target_date=target_date, ric_code='CMCU3', interval='1min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # =============================================
    # 1. ボリンジャーバンド戦略
    # =============================================
    logger.info("\n" + "=" * 60)
    logger.info("1. ボリンジャーバンド戦略")
    logger.info("=" * 60)

    bb_strategy = BollingerBandsStrategy(
        bb_period=20,
        bb_std=2.0,
        take_profit_pct=0.015,
        stop_loss_pct=0.015,
        max_positions=1
    )

    bb_result = bb_strategy.backtest(
        data=data,
        initial_capital=1000000,
        risk_per_trade=0.02
    )

    print_backtest_summary("ボリンジャーバンド戦略", bb_result)
    print_trade_details(bb_result['trades'])

    # =============================================
    # 2. ボラティリティブレイクアウト戦略
    # =============================================
    logger.info("\n" + "=" * 60)
    logger.info("2. ボラティリティブレイクアウト戦略")
    logger.info("=" * 60)

    vb_strategy = VolatilityBreakoutStrategy(
        atr_period=14,
        atr_multiplier=2.5,
        lookback_period=30,
        stop_loss_atr=2.0,
        take_profit_atr=4.0,
        max_positions=1
    )

    vb_result = vb_strategy.backtest(
        data=data,
        initial_capital=1000000,
        risk_per_trade=0.02
    )

    print_backtest_summary("ボラティリティブレイクアウト戦略", vb_result)
    print_trade_details(vb_result['trades'])

    # =============================================
    # 比較サマリー
    # =============================================
    print("\n" + "=" * 80)
    print("戦略比較サマリー")
    print("=" * 80)
    print(f"\n{'指標':<20} {'ボリンジャーバンド':<20} {'ボラティリティBO':<20}")
    print("-" * 80)
    print(f"{'トレード数':<20} {bb_result['total_trades']:>18}回  {vb_result['total_trades']:>18}回")
    print(f"{'勝率':<20} {bb_result['win_rate']:>17.2%}  {vb_result['win_rate']:>17.2%}")
    print(f"{'総リターン':<20} {bb_result['total_return']:>17.2%}  {vb_result['total_return']:>17.2%}")
    print(f"{'総損益':<20} {bb_result['total_pnl']:>15,.0f}円  {vb_result['total_pnl']:>15,.0f}円")
    print(f"{'取引コスト':<20} {bb_result.get('total_trading_costs', 0):>15,.0f}円  {vb_result.get('total_trading_costs', 0):>15,.0f}円")
    print(f"{'最大DD':<20} {bb_result['max_drawdown']:>17.2%}  {vb_result['max_drawdown']:>17.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()
