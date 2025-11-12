"""
LME Copper データ取得・DB格納スクリプト

使用方法:
    python fetch_lme_copper_data.py

環境変数:
    EIKON_API_KEY: Refinitiv Eikon APIキー
"""
import logging
from datetime import datetime, timedelta
import sys
import os

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.lme_client import LMEDataClient

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("LME Copper データ取得開始")
    logger.info("=" * 60)

    # APIキーの確認
    api_key = os.getenv('EIKON_API_KEY')
    if not api_key:
        # config.jsonから読み込みを試みる
        import json
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('eikon_api_key')

    if not api_key or api_key == 'YOUR_EIKON_API_KEY_HERE':
        logger.error("APIキーが設定されていません")
        logger.error("以下のいずれかの方法でAPIキーを設定してください:")
        logger.error("1. 環境変数 EIKON_API_KEY を設定")
        logger.error("2. config.json ファイルを作成（config.json.template を参考）")
        return

    # データ取得設定
    ric_codes = ['CMCU3']  # LME Copper 3M
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 直近1週間

    logger.info(f"取得期間: {start_date.date()} ～ {end_date.date()}")
    logger.info(f"RICコード: {', '.join(ric_codes)}")
    logger.info(f"インターバル: 1min")

    # クライアント初期化
    try:
        client = LMEDataClient(app_key=api_key, use_cache=True)
        client.connect()
        logger.info("Refinitiv API接続成功")
    except Exception as e:
        logger.error(f"API接続エラー: {e}")
        logger.error("Eikon Desktopアプリケーションが起動しているか確認してください")
        return

    try:
        # 各RICコードについてデータを取得
        for ric_code in ric_codes:
            logger.info("-" * 60)
            logger.info(f"RICコード: {ric_code} のデータ取得開始")

            data = client.get_lme_intraday_data(
                ric_code=ric_code,
                start_date=start_date,
                end_date=end_date,
                interval='1min'
            )

            if data is not None and not data.empty:
                logger.info(f"✓ {ric_code}: {len(data)}行のデータを取得・保存しました")
                logger.info(f"  期間: {data.index.min()} ～ {data.index.max()}")
                logger.info(f"  カラム: {list(data.columns)}")

                # サンプルデータを表示
                logger.info(f"\n  最新5行のデータ:")
                logger.info(f"\n{data.tail(5)}")
            else:
                logger.warning(f"✗ {ric_code}: データが取得できませんでした")

    except Exception as e:
        logger.error(f"データ取得エラー: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # クライアント切断
        client.disconnect()

    logger.info("=" * 60)
    logger.info("LME Copper データ取得完了")
    logger.info("=" * 60)

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
                COUNT(*) as total_records
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
                ric_code, interval, earliest, latest, count = row
                logger.info(f"RIC: {ric_code} ({interval})")
                logger.info(f"  期間: {earliest} ～ {latest}")
                logger.info(f"  レコード数: {count:,}行")
                logger.info("-" * 80)
        else:
            logger.info("データベースにデータが保存されていません")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"データベース確認エラー: {e}")


if __name__ == '__main__':
    main()
