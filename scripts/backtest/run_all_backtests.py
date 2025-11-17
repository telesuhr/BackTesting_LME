"""
全メタル×全戦略 統一バックテスト実行スクリプト

6メタル×4戦略の組み合わせを1つのコマンドで実行します。

使用方法:
    # 全メタル×全戦略を実行
    python scripts/backtest/run_all_backtests.py

    # 特定メタルのみ実行
    python scripts/backtest/run_all_backtests.py --metals copper zinc

    # 特定戦略のみ実行
    python scripts/backtest/run_all_backtests.py --strategies rsi bb_rsi

    # 組み合わせ指定
    python scripts/backtest/run_all_backtests.py --metals copper aluminium --strategies bollinger rsi
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import importlib
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.metals_config import (
    METALS_CONFIG,
    STRATEGIES_CONFIG,
    BACKTEST_CONFIG,
    DATABASE_CONFIG,
    OUTPUT_CONFIG,
    get_all_metals,
    get_all_strategies,
    get_metal_config,
    get_strategy_config
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(ric_code: str, start_date: str, end_date: str, interval: str = '15min') -> Optional[pd.DataFrame]:
    """データベースからデータを読み込む"""
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
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
            logger.error(f"データが見つかりません: {ric_code} ({start_date} - {end_date})")
            return None

        df.set_index('timestamp', inplace=True)

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


def create_strategy_instance(strategy_key: str):
    """
    戦略インスタンスを動的に作成

    config/metals_config.pyの設定から戦略クラスを動的にロードします。
    新しい戦略を追加する場合は、metals_config.pyのSTRATEGIES_CONFIGに
    追加するだけで自動的に利用可能になります（スクリプト修正不要）。

    例: 'rsi' → 'src.strategy.rsi_reversal.RSIReversalStrategy'
    """
    strategy_config = get_strategy_config(strategy_key)

    # モジュールとクラスを動的にインポート
    # importlibを使って文字列からモジュールとクラスをロード
    module_path = strategy_config['module']  # 例: 'src.strategy.rsi_reversal'
    class_name = strategy_config['class_name']  # 例: 'RSIReversalStrategy'

    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)

    # パラメーターを展開してインスタンス化
    params = strategy_config['params']
    return strategy_class(**params)


def save_visualization(
    data: pd.DataFrame,
    result: Dict[str, Any],
    strategy_instance: Any,
    metal_key: str,
    strategy_key: str
) -> str:
    """バックテスト結果を可視化して保存"""
    metal_config = get_metal_config(metal_key)
    strategy_config = get_strategy_config(strategy_key)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_CONFIG['base_dir'], metal_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{strategy_key}_backtest_1year_100mt_{timestamp}.png")

    trades = result['trades']

    # グラフを作成（3段）
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])  # 価格チャート
    ax2 = fig.add_subplot(gs[1, :])  # エクイティカーブ
    ax3 = fig.add_subplot(gs[2, 0])  # 月別損益
    ax4 = fig.add_subplot(gs[2, 1])  # 勝敗分布

    # === 上部: 価格チャート（サンプリング） ===
    sampled_data = data.iloc[::96]  # 15分×96 = 24時間分ごと
    ax1.plot(sampled_data.index, sampled_data['close'], color='black', linewidth=1, alpha=0.7)

    # トレードをプロット
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl = trade['pnl']

        # エントリーマーカー
        color = 'green' if trade['position'] == 1 else 'red'
        marker = '^' if trade['position'] == 1 else 'v'
        ax1.scatter(entry_time, entry_price, color=color, marker=marker, s=30, alpha=0.6, zorder=5)

        # エグジットマーカー
        exit_color = 'blue' if pnl > 0 else 'red'
        exit_marker = 'o' if pnl > 0 else 'x'
        ax1.scatter(exit_time, exit_price, color=exit_color, marker=exit_marker, s=30, alpha=0.6, zorder=5)

        # 大きな利益のトレードにラベル
        if abs(pnl) > 20000:
            ax1.annotate(f"${pnl/1000:.0f}k", xy=(exit_time, exit_price),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color='green' if pnl > 0 else 'red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    position_size = getattr(strategy_instance, 'fixed_position_size', 100.0)
    title = (f"LME {metal_config['name_en']} - {strategy_config['name']} Strategy "
            f"(15分足, {position_size:.0f}MT, 2024-11-11 ~ 2025-11-11)")

    ax1.set_ylabel('Price (USD/MT)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # === 中部: エクイティカーブ ===
    initial_capital = result['final_capital'] - result['total_pnl']
    equity_curve = [initial_capital]
    equity_times = [data.index[0]]

    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
        equity_times.append(trade['exit_time'])

    ax2.plot(equity_times, equity_curve, color='darkblue', linewidth=2.5)
    ax2.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Initial Capital')
    ax2.fill_between(equity_times, initial_capital, equity_curve, alpha=0.2, color='blue')

    # 最大ドローダウンをハイライト
    peak = equity_curve[0]
    max_dd_idx = 0
    max_dd = 0
    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_idx = i

    if max_dd_idx > 0:
        ax2.scatter(equity_times[max_dd_idx], equity_curve[max_dd_idx],
                   color='red', s=100, zorder=10, marker='v',
                   label=f'Max DD: {max_dd:.1%}')

    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Equity (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    # パフォーマンス統計
    stats_text = f"""【Strategy Parameters】
{strategy_config['name']}
Position: {position_size:.0f} MT

【Performance】
Total Return: {result['total_return']:.1%}
Total P&L: ${result['total_pnl']/1000:.1f}k
Win Rate: {result['win_rate']:.1%}
Max DD: {result['max_drawdown']:.1%}
Sharpe: {result['sharpe_ratio']:.2f}
Trades: {result['total_trades']}
Avg Win: ${result['avg_win']/1000:.1f}k
Avg Loss: ${result['avg_loss']/1000:.1f}k"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # === 左下: 月別損益 ===
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['exit_month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')

        monthly_pnl = trades_df.groupby('exit_month')['pnl'].sum()
        monthly_labels = [str(m) for m in monthly_pnl.index]

        colors = ['green' if pnl > 0 else 'red' for pnl in monthly_pnl.values]
        ax3.bar(range(len(monthly_pnl)), monthly_pnl.values / 1000, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(monthly_pnl)))
        ax3.set_xticklabels(monthly_labels, rotation=45, ha='right')

    ax3.set_ylabel('P&L ($1000)', fontsize=11, fontweight='bold')
    ax3.set_title('Monthly P&L', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=1)

    # === 右下: 勝敗分布 ===
    if trades:
        winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_pnls = [abs(t['pnl']) for t in trades if t['pnl'] <= 0]

        if winning_pnls or losing_pnls:
            ax4.hist([winning_pnls, losing_pnls], bins=20, color=['green', 'red'],
                    alpha=0.6, label=['Wins', 'Losses'])

    ax4.set_xlabel('P&L (USD)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Win/Loss Distribution', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_file, dpi=OUTPUT_CONFIG['graph_dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"グラフ保存完了: {output_file}")

    return output_file


def run_single_backtest(
    metal_key: str,
    strategy_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    単一のバックテストを実行

    Args:
        metal_key: メタルキー
        strategy_key: 戦略キー
        start_date: 開始日
        end_date: 終了日

    Returns:
        バックテスト結果（失敗時はNone）
    """
    try:
        metal_config = get_metal_config(metal_key)
        strategy_config = get_strategy_config(strategy_key)

        logger.info("=" * 60)
        logger.info(f"{metal_config['name']}（{metal_config['ric']}）× {strategy_config['name']}")
        logger.info("=" * 60)

        # デフォルト値設定
        if start_date is None:
            start_date = BACKTEST_CONFIG['start_date']
        if end_date is None:
            end_date = BACKTEST_CONFIG['end_date']

        # データ読み込み
        data = load_data(
            ric_code=metal_config['ric'],
            start_date=start_date,
            end_date=end_date,
            interval=BACKTEST_CONFIG['interval']
        )

        if data is None or data.empty:
            logger.error("データ読み込みに失敗しました")
            return None

        # 戦略インスタンス作成
        strategy = create_strategy_instance(strategy_key)

        # バックテスト実行
        result = strategy.backtest(
            data=data,
            initial_capital=BACKTEST_CONFIG['initial_capital'],
            risk_per_trade=BACKTEST_CONFIG['risk_per_trade']
        )

        # 可視化
        output_file = save_visualization(
            data=data,
            result=result,
            strategy_instance=strategy,
            metal_key=metal_key,
            strategy_key=strategy_key
        )

        # 結果に追加情報を付与
        result['metal_key'] = metal_key
        result['metal_name'] = metal_config['name']
        result['strategy_key'] = strategy_key
        result['strategy_name'] = strategy_config['name']
        result['output_file'] = output_file

        logger.info(f"✓ バックテスト完了: {metal_config['name']} × {strategy_config['name']}")
        logger.info(f"  リターン: {result['total_return']:.2%}")
        logger.info(f"  勝率: {result['win_rate']:.2%}")
        logger.info(f"  Sharpe: {result['sharpe_ratio']:.2f}")

        return result

    except Exception as e:
        logger.error(f"バックテストエラー: {metal_key} × {strategy_key}")
        logger.error(f"  エラー詳細: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_backtests(
    metals: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    全バックテストを実行

    Args:
        metals: 対象メタルリスト（Noneの場合は全メタル）
        strategies: 対象戦略リスト（Noneの場合は全戦略）
        start_date: 開始日
        end_date: 終了日

    Returns:
        全バックテスト結果のリスト
    """
    if metals is None:
        metals = get_all_metals()

    if strategies is None:
        strategies = get_all_strategies()

    logger.info("=" * 60)
    logger.info("全メタル×全戦略 バックテスト実行")
    logger.info("=" * 60)
    logger.info(f"対象メタル: {', '.join(metals)}")
    logger.info(f"対象戦略: {', '.join(strategies)}")
    logger.info(f"組み合わせ数: {len(metals)} × {len(strategies)} = {len(metals) * len(strategies)}")
    logger.info("=" * 60)

    results = []
    total_count = len(metals) * len(strategies)
    current_count = 0
    success_count = 0
    fail_count = 0

    for metal_key in metals:
        for strategy_key in strategies:
            current_count += 1
            logger.info(f"\n【進捗: {current_count}/{total_count}】")

            result = run_single_backtest(
                metal_key=metal_key,
                strategy_key=strategy_key,
                start_date=start_date,
                end_date=end_date
            )

            if result is not None:
                results.append(result)
                success_count += 1
            else:
                fail_count += 1

    # サマリー表示
    logger.info("\n" + "=" * 60)
    logger.info("全バックテスト完了サマリー")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count}/{total_count}組み合わせ")
    logger.info(f"失敗: {fail_count}/{total_count}組み合わせ")
    logger.info("=" * 60)

    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='全メタル×全戦略 統一バックテスト実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全メタル×全戦略を実行
  python scripts/backtest/run_all_backtests.py

  # 特定メタルのみ実行
  python scripts/backtest/run_all_backtests.py --metals copper zinc

  # 特定戦略のみ実行
  python scripts/backtest/run_all_backtests.py --strategies rsi bb_rsi

  # 組み合わせ指定
  python scripts/backtest/run_all_backtests.py --metals copper aluminium --strategies bollinger rsi
        """
    )

    parser.add_argument(
        '--metals',
        nargs='+',
        choices=get_all_metals(),
        help='対象メタル（デフォルト: 全メタル）'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=get_all_strategies(),
        help='対象戦略（デフォルト: 全戦略）'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='開始日（YYYY-MM-DD形式）'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='終了日（YYYY-MM-DD形式）'
    )

    args = parser.parse_args()

    # バックテスト実行
    results = run_all_backtests(
        metals=args.metals,
        strategies=args.strategies,
        start_date=args.start,
        end_date=args.end
    )

    logger.info(f"\n全バックテスト完了！結果: {len(results)}件")

    # サマリー生成
    if results:
        try:
            from generate_summary import generate_all_summaries
            logger.info("\nサマリー生成中...")
            summary_files = generate_all_summaries(results)
            logger.info("サマリー生成完了！")
        except Exception as e:
            logger.warning(f"サマリー生成に失敗しました: {e}")

    # 終了コード設定
    exit_code = 0 if results else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
