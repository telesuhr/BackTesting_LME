"""
ボリンジャーバンド+RSI組み合わせ戦略の1年間バックテスト可視化スクリプト（15分足）

使用方法:
    python visualize_backtest_1year.py
"""
import logging
from datetime import datetime
import sys
import os
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy.bollinger_rsi_combined import BollingerRSICombinedStrategy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォントを設定（macOSの場合）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_1year_data(start_date='2024-11-11', end_date='2025-11-11', ric_code='CMAL3', interval='15min'):
    """データベースから1年分のデータを読み込む"""
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

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def visualize_1year_backtest(data, result, strategy, strategy_name='bb_rsi', output_file=None):
    """
    1年間のバックテスト結果を可視化

    Args:
        data: 価格データ
        result: バックテスト結果
        strategy: 戦略インスタンス（パラメーター表示用）
        strategy_name: 戦略名（ファイル名の一部に使用）
        output_file: 出力ファイル名（Noneの場合、タイムスタンプ付きで自動生成）
    """
    # output_fileがNoneの場合、タイムスタンプ付きで生成
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'{strategy_name}_backtest_1year_100mt_{timestamp}.png'
    trades = result['trades']

    # グラフを作成（3段）
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])  # 価格チャート（全幅）
    ax2 = fig.add_subplot(gs[1, :])  # エクイティカーブ（全幅）
    ax3 = fig.add_subplot(gs[2, 0])  # 月別損益
    ax4 = fig.add_subplot(gs[2, 1])  # 勝敗分布

    # === 上部: 価格チャート（サンプリング） ===
    # データ量が多いので10日ごとにサンプリング
    sampled_data = data.iloc[::96]  # 15分×96 = 24時間分ごと

    ax1.plot(sampled_data.index, sampled_data['close'], color='black', linewidth=1, alpha=0.7)

    # トレードをプロット（エントリー/エグジットのみ）
    for i, trade in enumerate(trades):
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl = trade['pnl']

        # エントリーマーカー（小さく）
        color = 'green' if trade['position'] == 1 else 'red'
        marker = '^' if trade['position'] == 1 else 'v'
        ax1.scatter(entry_time, entry_price, color=color, marker=marker,
                  s=30, alpha=0.6, zorder=5)

        # エグジットマーカー（利益/損失で色分け）
        exit_color = 'blue' if pnl > 0 else 'red'
        exit_marker = 'o' if pnl > 0 else 'x'
        ax1.scatter(exit_time, exit_price, color=exit_color, marker=exit_marker,
                  s=30, alpha=0.6, zorder=5)

        # 大きな利益のトレードにラベル
        if abs(pnl) > 20000:
            ax1.annotate(f"${pnl/1000:.0f}k", xy=(exit_time, exit_price),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color='green' if pnl > 0 else 'red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # タイトルに戦略パラメーターを含める
    title = (f'LME Aluminium - Bollinger+RSI Combined Strategy (15分足, {strategy.fixed_position_size:.0f}MT, '
            f'BB{strategy.bb_period}/{strategy.bb_std}σ, 2024-11-11 ~ 2025-11-11)')

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
    ax2.axhline(y=initial_capital, color='gray', linestyle='--',
                linewidth=1.5, alpha=0.7, label='Initial Capital')
    ax2.fill_between(equity_times, initial_capital, equity_curve,
                     alpha=0.2, color='blue')

    # 最大ドローダウンをハイライト
    peak = equity_curve[0]
    max_dd_idx = 0
    max_dd = 0
    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
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

    # パフォーマンス統計をテキストボックスで表示
    stats_text = f"""【Strategy Parameters】
BB Period: {strategy.bb_period}
BB Std: {strategy.bb_std}σ
Position: {strategy.fixed_position_size:.0f} MT
Exit: ±2σ from entry

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
    winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_pnls = [abs(t['pnl']) for t in trades if t['pnl'] <= 0]

    ax4.hist([winning_pnls, losing_pnls], bins=20, color=['green', 'red'],
            alpha=0.6, label=['Wins', 'Losses'])
    ax4.set_xlabel('P&L (USD)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Win/Loss Distribution', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nグラフを保存しました: {output_file}")

    return output_file


def main():
    """メイン処理"""
    start_date = '2024-11-11'
    end_date = '2025-11-11'

    logger.info("=" * 60)
    logger.info(f"ボリンジャーバンド+RSI組み合わせ戦略 1年間可視化（15分足, 100MT）")
    logger.info("=" * 60)

    # データ読み込み
    data = load_1year_data(start_date=start_date, end_date=end_date, ric_code='CMAL3', interval='15min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # 戦略実行
    strategy = BollingerRSICombinedStrategy(
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        max_positions=1,
        fixed_position_size=100.0
    )

    result = strategy.backtest(
        data=data,
        initial_capital=100000.0,
        risk_per_trade=0.02
    )

    # 可視化
    output_file = visualize_1year_backtest(data, result, strategy, strategy_name='bb_rsi')

    print("\n可視化が完了しました。")
    print(f"生成されたファイル: {output_file}")


if __name__ == '__main__':
    main()
