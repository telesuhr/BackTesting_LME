"""
LME Copper 長期間データ取得スクリプト

大量のデータを月単位で分割して取得し、データベースに保存
"""
import logging
from datetime import datetime, timedelta
import sys
import os
import time

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.lme_client import LMEDataClient

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_data_by_month(client, ric_code, start_date, end_date, interval='1min'):
    """
    月単位でデータを取得

    Args:
        client: LMEDataClient
        ric_code: RICコード
        start_date: 開始日
        end_date: 終了日
        interval: データ間隔

    Returns:
        成功した月数
    """
    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date < end_date:
        # 1ヶ月間のデータを取得
        month_end = min(
            datetime(current_date.year, current_date.month, 1) + timedelta(days=32),
            end_date
        )
        month_end = datetime(month_end.year, month_end.month, 1) - timedelta(days=1)

        logger.info(f"\n{'='*60}")
        logger.info(f"期間: {current_date.date()} ～ {month_end.date()}")
        logger.info(f"{'='*60}")

        try:
            data = client.get_lme_intraday_data(
                ric_code=ric_code,
                start_date=current_date,
                end_date=month_end,
                interval=interval
            )

            if data is not None and not data.empty:
                logger.info(f"✓ {len(data)}行のデータを取得・保存しました")
                success_count += 1
            else:
                logger.warning(f"✗ データが取得できませんでした")
                fail_count += 1

            # API制限対策: 少し待機
            time.sleep(2)

        except Exception as e:
            logger.error(f"✗ エラーが発生しました: {e}")
            fail_count += 1
            time.sleep(5)  # エラー時は長めに待機

        # 次の月へ
        current_date = datetime(month_end.year, month_end.month, 1) + timedelta(days=32)
        current_date = datetime(current_date.year, current_date.month, 1)

    return success_count, fail_count


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("LME Copper 長期間データ取得開始")
    logger.info("=" * 60)

    # データ取得期間を設定（1年分）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    logger.info(f"取得期間: {start_date.date()} ～ {end_date.date()}")
    logger.info(f"RICコード: CMCU3")
    logger.info(f"インターバル: 1min")

    # APIキーの確認
    api_key = os.getenv('EIKON_API_KEY')
    if not api_key:
        import json
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('eikon_api_key') or config.get('refinitiv_api_key')

    if not api_key or api_key == 'YOUR_EIKON_API_KEY_HERE':
        logger.error("APIキーが設定されていません")
        return

    # クライアント初期化
    try:
        client = LMEDataClient(app_key=api_key, use_cache=True)
        client.connect()
        logger.info("✓ Refinitiv API接続成功")
    except Exception as e:
        logger.error(f"✗ API接続エラー: {e}")
        logger.error("Eikon Desktopアプリケーションが起動しているか確認してください")
        return

    try:
        # 月単位でデータを取得
        success, fail = fetch_data_by_month(
            client=client,
            ric_code='CMCU3',
            start_date=start_date,
            end_date=end_date,
            interval='1min'
        )

        logger.info("\n" + "=" * 60)
        logger.info("データ取得完了")
        logger.info("=" * 60)
        logger.info(f"成功: {success}ヶ月")
        logger.info(f"失敗: {fail}ヶ月")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client.disconnect()

    # データベース内容の確認
    logger.info("\nデータベース内容を確認します...")
    check_database_content()


def check_database_content():
    """データベース内容を確認"""
    import psycopg2

    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='lme_copper_db',
            user='postgres',
            password=''
        )

        cursor = conn.cursor()

        # データサマリーを取得
        cursor.execute("""
            SELECT
                ric_code,
                interval,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data,
                COUNT(*) as total_records,
                COUNT(DISTINCT DATE(timestamp)) as trading_days
            FROM lme_copper_intraday_data
            GROUP BY ric_code, interval
            ORDER BY ric_code, interval
        """)

        results = cursor.fetchall()

        if results:
            logger.info("\n" + "=" * 80)
            logger.info("データベース内容サマリー:")
            logger.info("=" * 80)
            for row in results:
                ric_code, interval, earliest, latest, count, days = row
                logger.info(f"RIC: {ric_code} ({interval})")
                logger.info(f"  期間: {earliest} ～ {latest}")
                logger.info(f"  レコード数: {count:,}行")
                logger.info(f"  取引日数: {days}日")
                logger.info("-" * 80)
        else:
            logger.info("データベースにデータが保存されていません")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"データベース確認エラー: {e}")


if __name__ == '__main__':
    main()
