"""
欠落期間（2025-03-29 ~ 2025-11-03）のデータを取得
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


def fetch_by_month(client, start_date_str, end_date_str):
    """月単位でデータを取得"""
    from datetime import datetime, timedelta
    import time

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    current_date = start_date
    total_rows_1min = 0
    total_rows_15min = 0

    while current_date < end_date:
        # 1ヶ月間の期間を計算
        month_end = min(
            datetime(current_date.year, current_date.month, 1) + timedelta(days=32),
            end_date
        )
        month_end = datetime(month_end.year, month_end.month, 1) - timedelta(days=1)

        logger.info(f"\n{'='*60}")
        logger.info(f"期間: {current_date.date()} ～ {month_end.date()}")
        logger.info(f"{'='*60}")

        try:
            # 1分足データを取得（自動的にDBに保存される）
            df_1min = client.get_lme_intraday_data(
                ric_code='CMCU3',
                start_date=current_date,
                end_date=month_end,
                interval='1min'
            )

            if df_1min is not None and not df_1min.empty:
                logger.info(f"✓ 1分足: {len(df_1min):,}行")
                total_rows_1min += len(df_1min)

                # 15分足にリサンプル
                df_15min = df_1min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(subset=['open', 'high', 'low', 'close'])

                # 15分足もDBに保存
                if not df_15min.empty and client.db_manager:
                    client.db_manager.save_intraday_data(df_15min, ric_code='CMCU3', interval='15min')
                    logger.info(f"✓ 15分足: {len(df_15min):,}行")
                    total_rows_15min += len(df_15min)
            else:
                logger.warning(f"✗ データが取得できませんでした")

            # API制限対策
            time.sleep(2)

        except Exception as e:
            logger.error(f"✗ エラー: {e}")
            time.sleep(5)

        # 次の月へ
        current_date = datetime(month_end.year, month_end.month, 1) + timedelta(days=32)
        current_date = datetime(current_date.year, current_date.month, 1)

    logger.info(f"\n総取得データ: 1分足={total_rows_1min:,}行, 15分足={total_rows_15min:,}行")


def main():
    """欠落期間のデータを取得"""
    start_date = '2025-03-29'
    end_date = '2025-11-04'

    logger.info("=" * 60)
    logger.info(f"欠落期間のデータ取得: {start_date} ～ {end_date}")
    logger.info("=" * 60)

    client = LMEDataClient()

    try:
        # API接続
        client.connect()

        # 月単位でデータ取得
        fetch_by_month(client, start_date, end_date)

    finally:
        # API切断
        client.disconnect()


if __name__ == '__main__':
    main()
