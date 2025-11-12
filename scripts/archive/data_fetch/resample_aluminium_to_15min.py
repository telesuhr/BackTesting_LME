"""
アルミニウムの1分足データを15分足にリサンプル
"""
import logging
import sys
import os
import pandas as pd
import psycopg2

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.lme_db_manager import LMEDatabaseManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resample_aluminium_to_15min(ric_code='CMAL3'):
    """
    アルミニウムの1分足データを15分足にリサンプル

    Args:
        ric_code: RICコード
    """
    try:
        # データベース接続
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='lme_copper_db',
            user='postgres',
            password=''
        )

        # 1分足データを読み込み
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM lme_copper_intraday_data
            WHERE ric_code = %s AND interval = '1min'
            ORDER BY timestamp
        """

        logger.info(f"{ric_code}: 1分足データを読み込み中...")
        df_1min = pd.read_sql(query, conn, params=(ric_code,))

        if df_1min.empty:
            logger.error(f"{ric_code}: 1分足データが見つかりません")
            conn.close()
            return

        logger.info(f"{ric_code}: {len(df_1min):,}行の1分足データを読み込みました")

        # インデックスをタイムスタンプに設定
        df_1min.set_index('timestamp', inplace=True)

        # 15分足にリサンプル
        logger.info(f"{ric_code}: 15分足にリサンプル中...")
        df_15min = df_1min.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open', 'high', 'low', 'close'])

        logger.info(f"{ric_code}: {len(df_15min):,}行の15分足データを作成しました")

        # データベースに保存
        db_manager = LMEDatabaseManager()
        db_manager.connect()

        saved_count = db_manager.save_lme_intraday_data(ric_code, df_15min, interval='15min')

        db_manager.disconnect()
        conn.close()

        logger.info(f"{ric_code}: {saved_count}行の15分足データを保存しました")

    except Exception as e:
        logger.error(f"エラー: {e}")


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("LMEアルミニウム 15分足リサンプル")
    logger.info("=" * 60)

    resample_aluminium_to_15min(ric_code='CMAL3')

    logger.info("=" * 60)
    logger.info("リサンプル完了")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
