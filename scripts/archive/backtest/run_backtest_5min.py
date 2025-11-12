"""
LME Copper 5分足バックテストスクリプト

使用方法:
    python run_backtest_5min.py
"""
import logging
from datetime import datetime
import sys
import os
import pandas as pd
import psycopg2

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy.bollinger_bands import BollingerBandsStrategy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_5min_data(target_date='2025-11-10', ric_code='CMCU3', interval='5min'):
    """
    データベースから5分足データを読み込む

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

        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)

        logger.info(f"5分足データ読み込み成功: {target_date}")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def print_backtest_summary(result):
    """バックテスト結果のサマリーを表示"""
    print("\n" + "=" * 80)
    print("ボリンジャーバンド戦略 - バックテスト結果（5分足）")
    print("=" * 80)

    print(f"\n【基本統計】")
    print(f"  総トレード数:     {result['total_trades']:>6}回")

    if result['total_trades'] == 0:
        print("  ※ トレードが発生しませんでした")
        print("\n" + "=" * 80)
        return

    print(f"  勝ちトレード:     {result.get('winning_trades', 0):>6}回")
    print(f"  負けトレード:     {result.get('losing_trades', 0):>6}回")
    print(f"  勝率:             {result['win_rate']:>6.2%}")

    print(f"\n【損益】")
    print(f"  初期資本:         {result['final_capital'] - result['total_pnl']:>12,.0f}円")
    print(f"  最終資本:         {result['final_capital']:>12,.0f}円")
    print(f"  総損益:           {result['total_pnl']:>12,.0f}円")
    print(f"  総リターン:       {result['total_return']:>12.2%}")

    if 'total_trading_costs' in result:
        print(f"\n【取引コスト】")
        print(f"  総取引コスト:     {result['total_trading_costs']:>12,.0f}円")
        if result['total_trades'] > 0:
            print(f"  平均コスト/回:    {result['total_trading_costs'] / result['total_trades']:>12,.0f}円")
            pnl_before_costs = result['total_pnl'] + result['total_trading_costs']
            print(f"  手数料控除前損益:   {pnl_before_costs:>12,.0f}円")

    if result['winning_trades'] > 0:
        print(f"\n【トレード統計】")
        print(f"  平均利益:         {result['avg_win']:>12,.0f}円")
        if result['losing_trades'] > 0:
            print(f"  平均損失:         {result['avg_loss']:>12,.0f}円")
            print(f"  損益比率:         {abs(result['avg_win'] / result['avg_loss']):>12.2f}")

    print(f"\n【リスク指標】")
    print(f"  最大ドローダウン: {result['max_drawdown']:>12.2%}")
    print(f"  シャープレシオ:   {result['sharpe_ratio']:>12.2f}")

    print("\n" + "=" * 80)


def print_trade_details(trades):
    """全トレードの詳細を表示"""
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

        pnl_gross = trade.get('pnl_gross', trade['pnl'])
        pnl_gross_str = f"{pnl_gross:>,.0f}円"
        if pnl_gross > 0:
            pnl_gross_str = f"+ {pnl_gross_str}"
        else:
            pnl_gross_str = f"- {abs(pnl_gross):>,.0f}円"

        costs = trade.get('trading_costs', 0)
        costs_str = f"{costs:>,.0f}円"

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
    logger.info(f"LME Copper 5分足バックテスト: {target_date}")
    logger.info("=" * 60)

    # データ読み込み
    data = load_5min_data(target_date=target_date, ric_code='CMCU3', interval='5min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # 戦略実行
    strategy = BollingerBandsStrategy(
        bb_period=20,
        bb_std=2.0,
        take_profit_pct=0.015,
        stop_loss_pct=0.015,
        max_positions=1
    )

    logger.info("\n戦略パラメータ:")
    logger.info(f"  BB期間:               {strategy.bb_period}")
    logger.info(f"  BB標準偏差:           {strategy.bb_std}σ")
    logger.info(f"  最大ポジション数:      {strategy.max_positions}")

    result = strategy.backtest(
        data=data,
        initial_capital=1000000,
        risk_per_trade=0.02
    )

    # 結果を表示
    print_backtest_summary(result)
    print_trade_details(result['trades'])


if __name__ == '__main__':
    main()
