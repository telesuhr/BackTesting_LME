"""
ボリンジャーバンド戦略の1ヶ月バックテスト可視化スクリプト（15分足）

使用方法:
    python visualize_backtest_1month.py
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

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy.bollinger_bands import BollingerBandsStrategy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォントを設定（macOSの場合）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_1month_data(start_date='2025-10-11', end_date='2025-11-11', ric_code='CMCU3', interval='15min'):
    """データベースから1ヶ月分のデータを読み込む"""
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


def visualize_bollinger_strategy(data, result, output_file='bollinger_backtest_1month_100mt.png'):
    """
    ボリンジャーバンド戦略を可視化

    Args:
        data: 価格データ
        result: バックテスト結果
        output_file: 出力ファイル名
    """
    # ボリンジャーバンドを計算
    strategy = BollingerBandsStrategy(bb_period=20, bb_std=2.0, fixed_position_size=100.0)
    upper_2sigma, middle, lower_2sigma = strategy.calculate_bollinger_bands(data)

    # 3σバンドを計算
    close = data['close']
    std = close.rolling(window=20).std()
    upper_3sigma = middle + (std * 3.0)
    lower_3sigma = middle - (std * 3.0)

    # トレード情報を抽出
    trades = result['trades']

    # グラフを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), gridspec_kw={'height_ratios': [3, 1]})

    # 上部: 価格とボリンジャーバンド
    ax1.plot(data.index, data['close'], label='Close Price', color='black', linewidth=2, zorder=5)

    # ボリンジャーバンド（2σ）
    ax1.plot(data.index, upper_2sigma, label='Upper 2σ', color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.plot(data.index, middle, label='Middle (MA20)', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(data.index, lower_2sigma, label='Lower 2σ', color='green', linewidth=1.5, linestyle='--', alpha=0.7)

    # ボリンジャーバンド（3σ）
    ax1.plot(data.index, upper_3sigma, label='Upper 3σ (Stop Loss)', color='darkred',
            linewidth=1, linestyle=':', alpha=0.5)
    ax1.plot(data.index, lower_3sigma, label='Lower 3σ (Stop Loss)', color='darkgreen',
            linewidth=1, linestyle=':', alpha=0.5)

    # バンド間を塗りつぶし
    ax1.fill_between(data.index, lower_2sigma, upper_2sigma, alpha=0.1, color='gray')

    # トレードをプロット
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position = trade['position']
        exit_reason = trade['exit_reason']
        pnl = trade['pnl']

        if position == 1:  # ロング
            # エントリー: 緑の上向き三角
            ax1.scatter(entry_time, entry_price, color='green', marker='^',
                      s=300, zorder=10, edgecolors='black', linewidths=2)

            # エグジット
            if exit_reason == 'take_profit':
                # 利食い: 青い下向き三角
                ax1.scatter(exit_time, exit_price, color='blue', marker='v',
                          s=300, zorder=10, edgecolors='black', linewidths=2)
            else:
                # 損切り: 赤い×印
                ax1.scatter(exit_time, exit_price, color='red', marker='x',
                          s=300, zorder=10, linewidths=4)

        else:  # ショート
            # エントリー: 赤の下向き三角
            ax1.scatter(entry_time, entry_price, color='red', marker='v',
                      s=300, zorder=10, edgecolors='black', linewidths=2)

            # エグジット
            if exit_reason == 'take_profit':
                # 利食い: 青い上向き三角
                ax1.scatter(exit_time, exit_price, color='blue', marker='^',
                          s=300, zorder=10, edgecolors='black', linewidths=2)
            else:
                # 損切り: 赤い×印
                ax1.scatter(exit_time, exit_price, color='red', marker='x',
                          s=300, zorder=10, linewidths=4)

        # エントリーからエグジットまで線を引く
        ax1.plot([entry_time, exit_time], [entry_price, exit_price],
               color='purple', linewidth=2, alpha=0.4, zorder=3)

        # P&Lをラベル表示
        mid_time = entry_time + (exit_time - entry_time) / 2
        mid_price = (entry_price + exit_price) / 2
        pnl_text = f"${pnl:,.0f}"
        color = 'green' if pnl > 0 else 'red'
        ax1.text(mid_time, mid_price, pnl_text, fontsize=11, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    # グラフの装飾（上部）
    ax1.set_ylabel('Price (USD/MT)', fontsize=14, fontweight='bold')
    ax1.set_title(f'LME Copper - Bollinger Bands Strategy (15分足, 100MT, 2025-10-11 ~ 2025-11-11)',
                fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 凡例を追加（トレードマーカーの説明）
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=14, markeredgecolor='black', linewidth=2, label='ロングエントリー'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=14, markeredgecolor='black', linewidth=2, label='ショートエントリー'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=14, markeredgecolor='black', linewidth=2, label='利食い（ロング）'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=14, markeredgecolor='black', linewidth=2, label='利食い（ショート）'),
        Line2D([0], [0], marker='x', color='red', markersize=14,
               markeredgewidth=3, linestyle='None', label='損切り'),
    ]
    ax1.legend(handles=custom_lines, loc='upper right', fontsize=11)

    # 下部: エクイティカーブ
    equity_curve = [result['final_capital'] - result['total_pnl']]  # 初期資本
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])

    equity_times = [data.index[0]] + [t['exit_time'] for t in trades]

    ax2.plot(equity_times, equity_curve, color='darkblue', linewidth=2.5, marker='o', markersize=6)
    ax2.axhline(y=result['final_capital'] - result['total_pnl'], color='gray', linestyle='--',
                linewidth=1.5, alpha=0.7, label='Initial Capital')
    ax2.fill_between(equity_times, result['final_capital'] - result['total_pnl'], equity_curve,
                     alpha=0.2, color='blue')

    # エクイティカーブの装飾
    ax2.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Equity (USD)', fontsize=14, fontweight='bold')
    ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=11)

    # パフォーマンス統計をテキストボックスで表示
    stats_text = f"""Total Return: {result['total_return']:.2%}
Total P&L: ${result['total_pnl']:,.2f}
Win Rate: {result['win_rate']:.1%}
Max Drawdown: {result['max_drawdown']:.2%}
Sharpe Ratio: {result['sharpe_ratio']:.2f}
Total Trades: {result['total_trades']}"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # X軸の日時フォーマット
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nグラフを保存しました: {output_file}")

    # 統計情報を表示
    print("\n" + "=" * 80)
    print("トレード統計（100MT）")
    print("=" * 80)
    print(f"総トレード数: {len(trades)}回")

    long_entries = sum(1 for t in trades if t['position'] == 1)
    short_entries = sum(1 for t in trades if t['position'] == -1)
    print(f"  ロングエントリー: {long_entries}回")
    print(f"  ショートエントリー: {short_entries}回")

    take_profits = sum(1 for t in trades if t['exit_reason'] == 'take_profit')
    stop_losses = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    print(f"\n利食い: {take_profits}回 ({take_profits/len(trades)*100:.1f}%)")
    print(f"損切り: {stop_losses}回 ({stop_losses/len(trades)*100:.1f}%)")

    total_pnl = sum(t['pnl'] for t in trades)
    print(f"\n総損益: ${total_pnl:,.2f}")
    print(f"平均損益: ${total_pnl/len(trades):,.2f}/回")
    print("=" * 80)


def main():
    """メイン処理"""
    start_date = '2025-10-11'
    end_date = '2025-11-11'

    logger.info("=" * 60)
    logger.info(f"ボリンジャーバンド戦略 可視化（15分足, 100MT）: {start_date} - {end_date}")
    logger.info("=" * 60)

    # データ読み込み
    data = load_1month_data(start_date=start_date, end_date=end_date, ric_code='CMCU3', interval='15min')

    if data is None or data.empty:
        logger.error("データの読み込みに失敗しました")
        return

    # 戦略実行
    strategy = BollingerBandsStrategy(
        bb_period=20,
        bb_std=2.0,
        take_profit_pct=0.015,
        stop_loss_pct=0.015,
        max_positions=1,
        fixed_position_size=100.0
    )

    result = strategy.backtest(
        data=data,
        initial_capital=100000.0,
        risk_per_trade=0.02
    )

    # 可視化
    visualize_bollinger_strategy(data, result, output_file='bollinger_backtest_1month_100mt.png')

    print("\n可視化が完了しました。")
    print("生成されたファイル: bollinger_backtest_1month_100mt.png")


if __name__ == '__main__':
    main()
