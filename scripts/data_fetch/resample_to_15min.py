"""
1分足データを15分足にリサンプルしてデータベースに保存

使用方法:
    python resample_to_15min.py
"""
import logging
import sys
import os
import pandas as pd
import psycopg2

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_1min_data(ric_code='CMCU3'):
    """1分足データを読み込む"""
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
              AND interval = '1min'
              AND open IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=(ric_code,))
        conn.close()

        if df.empty:
            logger.error("1分足データが見つかりません")
            return None

        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)

        logger.info(f"1分足データ読み込み成功")
        logger.info(f"  期間: {df.index.min()} ～ {df.index.max()}")
        logger.info(f"  データ数: {len(df):,}行")

        return df

    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        return None


def resample_to_15min(df):
    """1分足を15分足にリサンプル"""
    logger.info("15分足にリサンプル中...")

    # OHLCVデータをリサンプル
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # NaNを除外（データが存在しない期間）
    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])

    logger.info(f"リサンプル完了: {len(resampled):,}行")

    return resampled


def save_to_database(df, ric_code='CMCU3', interval='15min'):
    """15分足データをデータベースに保存"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='lme_copper_db',
            user='postgres',
            password=''
        )
        cursor = conn.cursor()

        # まず既存の15分足データを削除
        delete_query = """
            DELETE FROM lme_copper_intraday_data
            WHERE ric_code = %s AND interval = %s
        """
        cursor.execute(delete_query, (ric_code, interval))
        logger.info(f"既存の{interval}データを削除しました")

        # 新しいデータを挿入
        insert_query = """
            INSERT INTO lme_copper_intraday_data
            (ric_code, timestamp, open, high, low, close, volume, interval)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ric_code, timestamp, interval) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        inserted_count = 0
        for timestamp, row in df.iterrows():
            cursor.execute(insert_query, (
                ric_code,
                timestamp,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                interval
            ))
            inserted_count += 1

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"{interval}データを{inserted_count}行保存しました")
        return True

    except Exception as e:
        logger.error(f"データベース保存エラー: {e}")
        return False


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("1分足 → 15分足リサンプル処理")
    logger.info("=" * 60)

    # 1分足データを読み込む
    df_1min = load_1min_data(ric_code='CMCU3')

    if df_1min is None or df_1min.empty:
        logger.error("1分足データの読み込みに失敗しました")
        return

    # 15分足にリサンプル
    df_15min = resample_to_15min(df_1min)

    if df_15min.empty:
        logger.error("リサンプルに失敗しました")
        return

    # データベースに保存
    success = save_to_database(df_15min, ric_code='CMCU3', interval='15min')

    if success:
        logger.info("=" * 60)
        logger.info("リサンプル完了")
        logger.info("=" * 60)
        logger.info(f"15分足データ数: {len(df_15min):,}行")
        logger.info(f"期間: {df_15min.index.min()} ～ {df_15min.index.max()}")
    else:
        logger.error("データベースへの保存に失敗しました")


if __name__ == '__main__':
    main()
