"""
全メタルデータ統一取得スクリプト

6メタル（銅、アルミニウム、亜鉛、ニッケル、鉛、錫）のデータを
1つのコマンドで一括取得します。

使用方法:
    # 全メタル取得
    python scripts/data_fetch/fetch_all_metals.py

    # 特定メタルのみ取得
    python scripts/data_fetch/fetch_all_metals.py --metals copper zinc nickel

    # 期間指定
    python scripts/data_fetch/fetch_all_metals.py --start 2024-01-01 --end 2025-01-01
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from typing import List, Optional

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.metals_config import (
    METALS_CONFIG,
    BACKTEST_CONFIG,
    get_all_metals,
    get_metal_config
)
from src.data.lme_client import LMEDataClient
from src.data.lme_db_manager import LMEDatabaseManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_metal_data(
    metal_key: str,
    start_date: str,
    end_date: str,
    interval: str = '15min',
    force_refresh: bool = False
) -> bool:
    """
    指定メタルのデータを取得

    Args:
        metal_key: メタルキー（copper, aluminium等）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
        interval: データ間隔
        force_refresh: 強制再取得フラグ

    Returns:
        成功したかどうか
    """
    try:
        metal_config = get_metal_config(metal_key)
        ric_code = metal_config['ric']
        metal_name = metal_config['name']

        logger.info(f"=" * 60)
        logger.info(f"{metal_name}（{ric_code}）データ取得開始")
        logger.info(f"=" * 60)
        logger.info(f"期間: {start_date} ～ {end_date}")
        logger.info(f"間隔: {interval}")

        # データベースマネージャー初期化
        db_manager = LMEDatabaseManager()

        # キャッシュチェック（force_refreshがFalseの場合）
        if not force_refresh:
            existing_count = db_manager.get_data_count(
                ric_code=ric_code,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            if existing_count > 0:
                logger.info(f"データベースに{existing_count}件のデータが存在します（キャッシュ使用）")
                return True

        # APIクライアント初期化
        api_client = LMEDataClient()

        # 日付文字列をdatetimeに変換
        from datetime import datetime as dt
        start_dt = dt.strptime(start_date, '%Y-%m-%d')
        end_dt = dt.strptime(end_date, '%Y-%m-%d')

        # データ取得
        df = api_client.get_lme_intraday_data(
            ric_code=ric_code,
            start_date=start_dt,
            end_date=end_dt,
            interval=interval
        )

        if df is None or df.empty:
            logger.warning(f"{metal_name}のデータ取得に失敗しました")
            return False

        logger.info(f"取得データ: {len(df)}行")

        # データベースに保存
        saved_count = db_manager.save_data(
            df=df,
            ric_code=ric_code,
            interval=interval
        )

        logger.info(f"{metal_name}データ保存完了: {saved_count}件")
        logger.info(f"=" * 60)

        return True

    except Exception as e:
        logger.error(f"{metal_key}データ取得エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def fetch_all_metals_data(
    metals: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '15min',
    force_refresh: bool = False
) -> dict:
    """
    全メタルのデータを一括取得

    Args:
        metals: 取得対象メタルのリスト（Noneの場合は全メタル）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
        interval: データ間隔
        force_refresh: 強制再取得フラグ

    Returns:
        各メタルの取得結果
    """
    # デフォルト値設定
    if metals is None:
        metals = get_all_metals()

    if start_date is None:
        start_date = BACKTEST_CONFIG['start_date']

    if end_date is None:
        end_date = BACKTEST_CONFIG['end_date']

    logger.info("=" * 60)
    logger.info("全メタルデータ一括取得")
    logger.info("=" * 60)
    logger.info(f"対象メタル: {', '.join(metals)}")
    logger.info(f"期間: {start_date} ～ {end_date}")
    logger.info(f"間隔: {interval}")
    logger.info("=" * 60)

    results = {}
    success_count = 0
    fail_count = 0

    for metal_key in metals:
        logger.info(f"\n【{metal_key}】処理開始...")

        success = fetch_metal_data(
            metal_key=metal_key,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            force_refresh=force_refresh
        )

        results[metal_key] = success

        if success:
            success_count += 1
            logger.info(f"✓ {metal_key}: 成功")
        else:
            fail_count += 1
            logger.error(f"✗ {metal_key}: 失敗")

    # サマリー表示
    logger.info("\n" + "=" * 60)
    logger.info("データ取得完了サマリー")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count}/{len(metals)}メタル")
    logger.info(f"失敗: {fail_count}/{len(metals)}メタル")

    if fail_count > 0:
        logger.warning("失敗したメタル:")
        for metal_key, success in results.items():
            if not success:
                logger.warning(f"  - {metal_key}")

    logger.info("=" * 60)

    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='全メタルデータ一括取得スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全メタル取得（デフォルト期間）
  python scripts/data_fetch/fetch_all_metals.py

  # 特定メタルのみ取得
  python scripts/data_fetch/fetch_all_metals.py --metals copper zinc nickel

  # 期間指定
  python scripts/data_fetch/fetch_all_metals.py --start 2024-01-01 --end 2025-01-01

  # 強制再取得
  python scripts/data_fetch/fetch_all_metals.py --force-refresh
        """
    )

    parser.add_argument(
        '--metals',
        nargs='+',
        choices=get_all_metals(),
        help='取得対象メタル（デフォルト: 全メタル）'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='開始日（YYYY-MM-DD形式、デフォルト: config設定値）'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='終了日（YYYY-MM-DD形式、デフォルト: config設定値）'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='15min',
        choices=['1min', '5min', '15min', '30min', '1h', '1d'],
        help='データ間隔（デフォルト: 15min）'
    )

    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='キャッシュを無視して強制再取得'
    )

    args = parser.parse_args()

    # データ取得実行
    results = fetch_all_metals_data(
        metals=args.metals,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        force_refresh=args.force_refresh
    )

    # 終了コード設定（全て成功なら0、失敗があれば1）
    exit_code = 0 if all(results.values()) else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
