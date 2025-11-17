"""
バックテスト結果サマリー自動生成スクリプト

全メタル×全戦略のバックテスト結果を集計し、
CSV/Markdown形式で出力します。

使用方法:
    # run_all_backtests.pyの結果をインポートして使用
    python scripts/backtest/generate_summary.py

    # または直接実行（既存の結果を読み込み）
    python scripts/backtest/generate_summary.py --scan-outputs
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import json
import re

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.metals_config import (
    METALS_CONFIG,
    STRATEGIES_CONFIG,
    OUTPUT_CONFIG,
    get_all_metals,
    get_all_strategies
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_summary_csv(results: List[Dict[str, Any]], output_dir: str = None, output_file: str = None) -> str:
    """
    結果をCSV形式で保存

    Args:
        results: バックテスト結果のリスト
        output_dir: 出力先ディレクトリ（日時フォルダのパス）
        output_file: 出力ファイル名（Noneの場合は自動生成）

    Returns:
        保存したファイルパス
    """
    if output_file is None:
        if output_dir is None:
            output_dir = OUTPUT_CONFIG['summary_dir']
        else:
            # 日時フォルダ配下にsummaryフォルダを作成
            output_dir = os.path.join(output_dir, 'summary')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'performance_summary.csv')

    # DataFrame作成
    df = pd.DataFrame([
        {
            'Metal': r['metal_name'],
            'Metal_Key': r['metal_key'],
            'Strategy': r['strategy_name'],
            'Strategy_Key': r['strategy_key'],
            'Total_Return_%': round(r['total_return'] * 100, 2),
            'Total_PnL_USD': round(r['total_pnl'], 2),
            'Final_Capital_USD': round(r['final_capital'], 2),
            'Win_Rate_%': round(r['win_rate'] * 100, 2),
            'Total_Trades': r['total_trades'],
            'Winning_Trades': r['winning_trades'],
            'Losing_Trades': r['losing_trades'],
            'Avg_Win_USD': round(r['avg_win'], 2),
            'Avg_Loss_USD': round(r['avg_loss'], 2),
            'Max_Drawdown_%': round(r['max_drawdown'] * 100, 2),
            'Sharpe_Ratio': round(r['sharpe_ratio'], 2),
            'Total_Trading_Costs_USD': round(r.get('total_trading_costs', 0), 2)
        }
        for r in results
    ])

    # リターンで降順ソート
    df = df.sort_values('Total_Return_%', ascending=False)

    # CSV保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"CSV保存完了: {output_file}")

    return output_file


def generate_summary_markdown(results: List[Dict[str, Any]], output_dir: str = None, output_file: str = None) -> str:
    """
    結果をMarkdown形式で保存

    Args:
        results: バックテスト結果のリスト
        output_dir: 出力先ディレクトリ（日時フォルダのパス）
        output_file: 出力ファイル名（Noneの場合は自動生成）

    Returns:
        保存したファイルパス
    """
    if output_file is None:
        if output_dir is None:
            output_dir = OUTPUT_CONFIG['summary_dir']
        else:
            # 日時フォルダ配下にsummaryフォルダを作成
            output_dir = os.path.join(output_dir, 'summary')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'performance_summary.md')

    # リターンで降順ソート
    sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)

    # Markdown生成
    md_lines = []
    md_lines.append("# LMEメタル取引バックテスト結果サマリー")
    md_lines.append("")
    md_lines.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"**総組み合わせ数**: {len(results)}件")
    md_lines.append("")

    # === 全体サマリーテーブル ===
    md_lines.append("## 全体サマリー（リターン降順）")
    md_lines.append("")
    md_lines.append("| メタル | 戦略 | リターン | 勝率 | Sharpe | Max DD | トレード数 |")
    md_lines.append("|--------|------|----------|------|--------|--------|-----------|")

    for r in sorted_results:
        return_str = f"{r['total_return']:.1%}"
        if r['total_return'] > 0.5:  # 50%以上
            return_str = f"**{return_str}** 🏆"
        elif r['total_return'] > 0.2:  # 20%以上
            return_str = f"**{return_str}**"

        md_lines.append(
            f"| {r['metal_name']} | {r['strategy_name']} | {return_str} | "
            f"{r['win_rate']:.1%} | {r['sharpe_ratio']:.2f} | {r['max_drawdown']:.1%} | "
            f"{r['total_trades']}回 |"
        )

    md_lines.append("")

    # === メタル別サマリー ===
    md_lines.append("## メタル別パフォーマンス")
    md_lines.append("")

    for metal_key in get_all_metals():
        metal_name = METALS_CONFIG[metal_key]['name']
        metal_results = [r for r in results if r['metal_key'] == metal_key]

        if not metal_results:
            continue

        md_lines.append(f"### {metal_name} ({metal_key})")
        md_lines.append("")
        md_lines.append("| 戦略 | リターン | 勝率 | Sharpe | Max DD | P&L |")
        md_lines.append("|------|----------|------|--------|--------|-----|")

        # リターン降順
        metal_results_sorted = sorted(metal_results, key=lambda x: x['total_return'], reverse=True)

        for r in metal_results_sorted:
            md_lines.append(
                f"| {r['strategy_name']} | {r['total_return']:.1%} | "
                f"{r['win_rate']:.1%} | {r['sharpe_ratio']:.2f} | "
                f"{r['max_drawdown']:.1%} | ${r['total_pnl']/1000:.1f}k |"
            )

        md_lines.append("")

    # === 戦略別サマリー ===
    md_lines.append("## 戦略別パフォーマンス")
    md_lines.append("")

    for strategy_key in get_all_strategies():
        strategy_name = STRATEGIES_CONFIG[strategy_key]['name']
        strategy_results = [r for r in results if r['strategy_key'] == strategy_key]

        if not strategy_results:
            continue

        md_lines.append(f"### {strategy_name} ({strategy_key})")
        md_lines.append("")
        md_lines.append("| メタル | リターン | 勝率 | Sharpe | Max DD | P&L |")
        md_lines.append("|--------|----------|------|--------|--------|-----|")

        # リターン降順
        strategy_results_sorted = sorted(strategy_results, key=lambda x: x['total_return'], reverse=True)

        for r in strategy_results_sorted:
            md_lines.append(
                f"| {r['metal_name']} | {r['total_return']:.1%} | "
                f"{r['win_rate']:.1%} | {r['sharpe_ratio']:.2f} | "
                f"{r['max_drawdown']:.1%} | ${r['total_pnl']/1000:.1f}k |"
            )

        md_lines.append("")

    # === トップパフォーマー ===
    md_lines.append("## トップパフォーマー（Top 10）")
    md_lines.append("")
    md_lines.append("| ランク | メタル | 戦略 | リターン | 勝率 | Sharpe | Max DD |")
    md_lines.append("|--------|--------|------|----------|------|--------|--------|")

    for i, r in enumerate(sorted_results[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        md_lines.append(
            f"| {i} {emoji} | {r['metal_name']} | {r['strategy_name']} | "
            f"{r['total_return']:.1%} | {r['win_rate']:.1%} | "
            f"{r['sharpe_ratio']:.2f} | {r['max_drawdown']:.1%} |"
        )

    md_lines.append("")

    # === 統計サマリー ===
    md_lines.append("## 統計サマリー")
    md_lines.append("")

    all_returns = [r['total_return'] for r in results]
    all_sharpes = [r['sharpe_ratio'] for r in results]
    all_win_rates = [r['win_rate'] for r in results]

    positive_returns = [r for r in results if r['total_return'] > 0]
    negative_returns = [r for r in results if r['total_return'] <= 0]

    md_lines.append(f"- **平均リターン**: {sum(all_returns) / len(all_returns):.2%}")
    md_lines.append(f"- **最大リターン**: {max(all_returns):.2%}")
    md_lines.append(f"- **最小リターン**: {min(all_returns):.2%}")
    md_lines.append(f"- **平均Sharpe**: {sum(all_sharpes) / len(all_sharpes):.2f}")
    md_lines.append(f"- **平均勝率**: {sum(all_win_rates) / len(all_win_rates):.1%}")
    md_lines.append(f"- **プラスリターン**: {len(positive_returns)}/{len(results)}組み合わせ ({len(positive_returns)/len(results):.1%})")
    md_lines.append(f"- **マイナスリターン**: {len(negative_returns)}/{len(results)}組み合わせ ({len(negative_returns)/len(results):.1%})")
    md_lines.append("")

    # ファイル保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    logger.info(f"Markdown保存完了: {output_file}")

    return output_file


def generate_all_summaries(results: List[Dict[str, Any]], output_dir: str = None) -> Dict[str, str]:
    """
    全形式のサマリーを生成

    Args:
        results: バックテスト結果のリスト
        output_dir: 出力先ディレクトリ（日時フォルダのパス）

    Returns:
        生成されたファイルのパス辞書
    """
    logger.info("=" * 60)
    logger.info("サマリー生成開始")
    logger.info("=" * 60)
    logger.info(f"対象結果数: {len(results)}件")
    if output_dir:
        logger.info(f"出力先: {output_dir}/summary/")

    output_files = {}

    # CSV生成
    logger.info("\nCSV生成中...")
    csv_file = generate_summary_csv(results, output_dir=output_dir)
    output_files['csv'] = csv_file

    # Markdown生成
    logger.info("\nMarkdown生成中...")
    md_file = generate_summary_markdown(results, output_dir=output_dir)
    output_files['markdown'] = md_file

    logger.info("\n" + "=" * 60)
    logger.info("サマリー生成完了")
    logger.info("=" * 60)
    logger.info(f"CSV: {csv_file}")
    logger.info(f"Markdown: {md_file}")
    logger.info("=" * 60)

    return output_files


def main():
    """メイン処理（スタンドアロン実行用）"""
    parser = argparse.ArgumentParser(
        description='バックテスト結果サマリー生成',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--scan-outputs',
        action='store_true',
        help='outputsフォルダから結果をスキャン（未実装）'
    )

    args = parser.parse_args()

    if args.scan_outputs:
        logger.error("--scan-outputs機能は未実装です")
        logger.info("run_all_backtests.pyから直接呼び出してください")
        sys.exit(1)
    else:
        logger.info("このスクリプトは run_all_backtests.py から呼び出されます")
        logger.info("直接実行する場合は --scan-outputs オプションを使用してください（未実装）")
        sys.exit(0)


if __name__ == '__main__':
    main()
