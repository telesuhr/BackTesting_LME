"""
LME Copper 1ヶ月間バックテストスクリプト（15分足）

使用方法:
    python run_backtest_1month.py
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

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_1month_data(start_date='2024-10-11', end_date='2024-11-11', ric_code='CMCU3', interval='15min'):
    """
    データベースから1ヶ月分のデータを読み込む

    Args:
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
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
              AND timestamp < %s::timestamp
              AND open IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=(ric_code, interval, start_date, end_date))
        conn.close()

        if df.empty:
            logger.error(f"データが見つかりません: {start_date} - {end_date}")
            return None

        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)

        logger.info(f"データ読み込み成功: {start_date} - {end_date}")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        # 取引日数を計算
        trading_days = df.index.date
        unique_days = pd.Series(trading_days).nunique()
        logger.info(f"  取引日数: {unique_days}日")

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def print_backtest_summary(result, start_date, end_date):
    """バックテスト結果のサマリーを表示"""
    print("\n" + "=" * 80)
    print(f"ボリンジャーバンド戦略 - 1ヶ月バックテスト結果（15分足）")
    print(f"期間: {start_date} ～ {end_date}")
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
    print(f"  初期資本:         ${result['final_capital'] - result['total_pnl']:>12,.2f}")
    print(f"  最終資本:         ${result['final_capital']:>12,.2f}")
    print(f"  総損益:           ${result['total_pnl']:>12,.2f}")
    print(f"  総リターン:       {result['total_return']:>12.2%}")

    if 'total_trading_costs' in result:
        print(f"\n【取引コスト】")
        print(f"  総取引コスト:     ${result['total_trading_costs']:>12,.2f}")
        if result['total_trades'] > 0:
            print(f"  平均コスト/回:    ${result['total_trading_costs'] / result['total_trades']:>12,.2f}")
            pnl_before_costs = result['total_pnl'] + result['total_trading_costs']
            print(f"  手数料控除前損益:   ${pnl_before_costs:>12,.2f}")

    if result['winning_trades'] > 0:
        print(f"\n【トレード統計】")
        print(f"  平均利益:         ${result['avg_win']:>12,.2f}")
        if result['losing_trades'] > 0:
            print(f"  平均損失:         ${result['avg_loss']:>12,.2f}")
            print(f"  損益比率:         {abs(result['avg_win'] / result['avg_loss']):>12.2f}")

    print(f"\n【リスク指標】")
    print(f"  最大ドローダウン: {result['max_drawdown']:>12.2%}")
    print(f"  シャープレシオ:   {result['sharpe_ratio']:>12.2f}")

    print("\n" + "=" * 80)


def print_trade_summary(trades, max_display=20):
    """トレードのサマリーを表示"""
    if len(trades) == 0:
        print("\n【トレード詳細】")
        print("  トレードが発生しませんでした")
        return

    print(f"\n【トレードサマリー】")
    print(f"  総トレード数: {len(trades)}回")

    # 勝ち/負けの統計
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    print(f"  勝ちトレード: {len(winning_trades)}回")
    print(f"  負けトレード: {len(losing_trades)}回")

    # 決済理由の統計
    take_profit_count = sum(1 for t in trades if t['exit_reason'] == 'take_profit')
    stop_loss_count = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    print(f"\n  利食い: {take_profit_count}回 ({take_profit_count/len(trades)*100:.1f}%)")
    print(f"  損切り: {stop_loss_count}回 ({stop_loss_count/len(trades)*100:.1f}%)")

    # 最初と最後のトレードを表示
    print(f"\n【最初の{min(5, len(trades))}件と最後の{min(5, len(trades))}件】")
    print("-" * 160)
    print(f"{'#':<4} {'エントリー時刻':<20} {'エグジット時刻':<20} {'ポジション':<8} "
          f"{'エントリー価格':<12} {'エグジット価格':<12} {'総損益':<15} {'手数料':<12} {'純損益':<15} {'理由':<12}")
    print("-" * 160)

    # 最初の5件
    for i, trade in enumerate(trades[:min(5, len(trades))], 1):
        print_trade_line(i, trade)

    if len(trades) > 10:
        print("  " + "." * 140)

    # 最後の5件
    if len(trades) > 5:
        start_idx = max(6, len(trades) - 4)
        for i, trade in enumerate(trades[-5:], start_idx):
            print_trade_line(i, trade)

    print("-" * 160)


def print_trade_line(i, trade):
    """1行のトレード情報を表示"""
    position_str = 'ロング' if trade['position'] == 1 else 'ショート'

    pnl_gross = trade.get('pnl_gross', trade['pnl'])
    pnl_gross_str = f"${pnl_gross:>,.2f}"
    if pnl_gross > 0:
        pnl_gross_str = f"+ {pnl_gross_str}"
    else:
        pnl_gross_str = f"- ${abs(pnl_gross):>,.2f}"

    costs = trade.get('trading_costs', 0)
    costs_str = f"${costs:>,.2f}"

    pnl_str = f"${trade['pnl']:>,.2f}"
    if trade['pnl'] > 0:
        pnl_str = f"+ {pnl_str}"
    else:
        pnl_str = f"- ${abs(trade['pnl']):>,.2f}"

    print(f"{i:<4} {str(trade['entry_time']):<20} {str(trade['exit_time']):<20} "
          f"{position_str:<8} {trade['entry_price']:>10,.2f} {trade['exit_price']:>10,.2f} "
          f"{pnl_gross_str:<15} {costs_str:<12} {pnl_str:<15} {trade['exit_reason']:<12}")


def main():
    """メイン処理"""
    # 1ヶ月分（2025-10-11から2025-11-11まで）
    start_date = '2025-10-11'
    end_date = '2025-11-11'

    logger.info("=" * 60)
    logger.info(f"LME Copper 1ヶ月間バックテスト（15分足）")
    logger.info(f"期間: {start_date} ～ {end_date}")
    logger.info("=" * 60)

    # データ読み込み
    data = load_1month_data(start_date=start_date, end_date=end_date, ric_code='CMCU3', interval='15min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # 戦略実行（ポジションサイズ: 100MT）
    strategy = BollingerBandsStrategy(
        bb_period=20,
        bb_std=2.0,
        take_profit_pct=0.015,
        stop_loss_pct=0.015,
        max_positions=1,
        fixed_position_size=100.0  # 100MT
    )

    logger.info("\n戦略パラメータ:")
    logger.info(f"  BB期間:               {strategy.bb_period}")
    logger.info(f"  BB標準偏差:           {strategy.bb_std}σ")
    logger.info(f"  ポジションサイズ:      {strategy.fixed_position_size:,.1f} MT")

    result = strategy.backtest(
        data=data,
        initial_capital=100000.0,  # 100,000 USD
        risk_per_trade=0.02
    )

    # 結果を表示
    print_backtest_summary(result, start_date, end_date)
    print_trade_summary(result['trades'])


if __name__ == '__main__':
    main()
