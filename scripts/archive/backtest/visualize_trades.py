"""
ボリンジャーバンド戦略の可視化スクリプト

使用方法:
    python visualize_trades.py
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


def load_single_day_data(target_date='2025-11-10', ric_code='CMCU3', interval='1min'):
    """データベースから特定日のデータを読み込む"""
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

        logger.info(f"データ読み込み成功: {target_date}")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        return df

    except Exception as e:
        logger.error(f"データベース読み込みエラー: {e}")
        return None


def visualize_bollinger_strategy(data, result, output_file='bollinger_trades.png'):
    """
    ボリンジャーバンド戦略を可視化

    Args:
        data: 価格データ
        result: バックテスト結果
        output_file: 出力ファイル名
    """
    # ボリンジャーバンドを計算
    strategy = BollingerBandsStrategy(bb_period=20, bb_std=2.0)
    upper_2sigma, middle, lower_2sigma = strategy.calculate_bollinger_bands(data)

    # 3σバンドを計算
    close = data['close']
    std = close.rolling(window=20).std()
    upper_3sigma = middle + (std * 3.0)
    lower_3sigma = middle - (std * 3.0)

    # トレード情報を抽出
    trades = result['trades']

    # グラフを作成
    fig, ax = plt.subplots(figsize=(20, 10))

    # 価格推移
    ax.plot(data.index, data['close'], label='Close Price', color='black', linewidth=1.5, zorder=5)

    # ボリンジャーバンド（2σ）
    ax.plot(data.index, upper_2sigma, label='Upper 2σ', color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax.plot(data.index, middle, label='Middle (MA20)', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(data.index, lower_2sigma, label='Lower 2σ', color='green', linewidth=1, linestyle='--', alpha=0.7)

    # ボリンジャーバンド（3σ）
    ax.plot(data.index, upper_3sigma, label='Upper 3σ (Stop Loss)', color='darkred',
            linewidth=1, linestyle=':', alpha=0.5)
    ax.plot(data.index, lower_3sigma, label='Lower 3σ (Stop Loss)', color='darkgreen',
            linewidth=1, linestyle=':', alpha=0.5)

    # バンド間を塗りつぶし
    ax.fill_between(data.index, lower_2sigma, upper_2sigma, alpha=0.1, color='gray')

    # トレードをプロット
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position = trade['position']
        exit_reason = trade['exit_reason']

        if position == 1:  # ロング
            # エントリー: 緑の上向き三角
            ax.scatter(entry_time, entry_price, color='green', marker='^',
                      s=150, zorder=10, edgecolors='black', linewidths=1)

            # エグジット
            if exit_reason == 'take_profit':
                # 利食い: 青い下向き三角
                ax.scatter(exit_time, exit_price, color='blue', marker='v',
                          s=150, zorder=10, edgecolors='black', linewidths=1)
            else:
                # 損切り: 赤い×印
                ax.scatter(exit_time, exit_price, color='red', marker='x',
                          s=150, zorder=10, linewidths=3)

        else:  # ショート
            # エントリー: 赤の下向き三角
            ax.scatter(entry_time, entry_price, color='red', marker='v',
                      s=150, zorder=10, edgecolors='black', linewidths=1)

            # エグジット
            if exit_reason == 'take_profit':
                # 利食い: 青い上向き三角
                ax.scatter(exit_time, exit_price, color='blue', marker='^',
                          s=150, zorder=10, edgecolors='black', linewidths=1)
            else:
                # 損切り: 赤い×印
                ax.scatter(exit_time, exit_price, color='red', marker='x',
                          s=150, zorder=10, linewidths=3)

        # エントリーからエグジットまで線を引く
        ax.plot([entry_time, exit_time], [entry_price, exit_price],
               color='purple', linewidth=1, alpha=0.3, zorder=3)

    # グラフの装飾
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD/MT)', fontsize=14, fontweight='bold')
    ax.set_title('LME Copper - Bollinger Bands Strategy (2025-11-10)',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # X軸の日時フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)

    # 凡例を追加（トレードマーカーの説明）
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, markeredgecolor='black', label='ロングエントリー'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='black', label='ショートエントリー'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='black', label='利食い（ロング）'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='black', label='利食い（ショート）'),
        Line2D([0], [0], marker='x', color='red', markersize=10,
               markeredgewidth=2, linestyle='None', label='損切り'),
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nグラフを保存しました: {output_file}")

    # 統計情報を表示
    print("\n" + "=" * 80)
    print("トレード統計")
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
    print("=" * 80)


def main():
    """メイン処理"""
    target_date = '2025-11-10'

    logger.info("=" * 60)
    logger.info(f"ボリンジャーバンド戦略 可視化: {target_date}")
    logger.info("=" * 60)

    # データ読み込み
    data = load_single_day_data(target_date=target_date, ric_code='CMCU3', interval='1min')

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

    result = strategy.backtest(
        data=data,
        initial_capital=1000000,
        risk_per_trade=0.02
    )

    # 可視化
    visualize_bollinger_strategy(data, result, output_file='bollinger_trades_20251110.png')

    print("\n可視化が完了しました。")
    print("生成されたファイル: bollinger_trades_20251110.png")


if __name__ == '__main__':
    main()
