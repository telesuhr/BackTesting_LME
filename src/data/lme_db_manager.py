"""
LME専用 PostgreSQLデータベース管理クラス
"""
import psycopg2
import psycopg2.extras
import pandas as pd
import logging
from datetime import datetime
from typing import Optional
import os


logger = logging.getLogger(__name__)


class LMEDatabaseManager:
    """LME専用 PostgreSQLデータベース管理"""

    def __init__(self, config: dict = None):
        """
        Args:
            config: データベース接続設定辞書
                   Noneの場合は環境変数から読み込む
        """
        if config is None:
            config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'lme_copper_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }

        self.config = config
        self.conn = None

    def connect(self):
        """データベースに接続"""
        try:
            self.conn = psycopg2.connect(**self.config)
            logger.info(f"データベース接続成功: {self.config['database']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"データベース接続エラー: {e}")
            return False

    def disconnect(self):
        """データベース接続を切断"""
        if self.conn:
            self.conn.close()
            logger.info("データベース切断完了")

    def save_lme_intraday_data(
        self,
        ric_code: str,
        data: pd.DataFrame,
        interval: str = '1min'
    ) -> int:
        """
        LME分足データをデータベースに保存

        Args:
            ric_code: RICコード（例: 'CMCU3'）
            data: 分足データ（DatetimeIndexを持つDataFrame）
            interval: データ間隔

        Returns:
            保存した行数
        """
        if data.empty:
            return 0

        cursor = self.conn.cursor()
        inserted_count = 0

        try:
            for timestamp, row in data.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO lme_copper_intraday_data
                        (ric_code, timestamp, open, high, low, close, volume, interval)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ric_code, timestamp, interval) DO NOTHING
                    """, (
                        ric_code,
                        timestamp,
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None,
                        interval
                    ))
                    inserted_count += cursor.rowcount
                except Exception as e:
                    logger.warning(f"行挿入エラー ({timestamp}): {e}")
                    continue

            self.conn.commit()
            logger.info(f"{ric_code}: {inserted_count}行をDBに保存")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"データ保存エラー: {e}")
            raise
        finally:
            cursor.close()

        return inserted_count

    def get_lme_intraday_data(
        self,
        ric_code: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1min'
    ) -> Optional[pd.DataFrame]:
        """
        LME分足データをデータベースから取得

        Args:
            ric_code: RICコード
            start_date: 開始日時
            end_date: 終了日時
            interval: データ間隔

        Returns:
            分足データのDataFrame、データがない場合はNone
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM lme_copper_intraday_data
                WHERE ric_code = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                  AND interval = %s
                ORDER BY timestamp
            """, (ric_code, start_date, end_date, interval))

            rows = cursor.fetchall()

            if not rows:
                return None

            # DataFrameに変換
            df = pd.DataFrame(
                rows,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df.set_index('timestamp', inplace=True)

            # Decimal型をfloat型に変換
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            # volumeはintに変換（NULL値は0に）
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0).astype(int)

            logger.info(f"{ric_code}: DBから{len(df)}行を取得")
            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None
        finally:
            cursor.close()

    def log_lme_fetch(
        self,
        ric_code: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        source: str,
        records_count: int
    ):
        """
        LMEデータ取得ログを記録

        Args:
            ric_code: RICコード
            start_date: 開始日時
            end_date: 終了日時
            interval: データ間隔
            source: データソース ('api' or 'cache')
            records_count: 取得レコード数
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO lme_data_fetch_log
                (ric_code, start_date, end_date, interval, source, records_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (ric_code, start_date, end_date, interval, source, records_count))

            self.conn.commit()

        except Exception as e:
            logger.warning(f"ログ記録エラー: {e}")
            self.conn.rollback()
        finally:
            cursor.close()

    def get_lme_cached_date_range(
        self,
        ric_code: str,
        interval: str = '1min'
    ) -> Optional[tuple]:
        """
        キャッシュされているデータの日付範囲を取得

        Args:
            ric_code: RICコード
            interval: データ間隔

        Returns:
            (最小日時, 最大日時) のタプル、データがない場合はNone
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM lme_copper_intraday_data
                WHERE ric_code = %s AND interval = %s
            """, (ric_code, interval))

            result = cursor.fetchone()

            if result[0] is None:
                return None

            return result

        except Exception as e:
            logger.error(f"日付範囲取得エラー: {e}")
            return None
        finally:
            cursor.close()

    def get_data_count(
        self,
        ric_code: str,
        interval: str,
        start_date: str = None,
        end_date: str = None
    ) -> int:
        """
        指定期間のデータ件数を取得

        Args:
            ric_code: RICコード
            interval: データ間隔
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）

        Returns:
            データ件数
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        try:
            if start_date and end_date:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM lme_copper_intraday_data
                    WHERE ric_code = %s
                      AND interval = %s
                      AND timestamp >= %s::timestamp
                      AND timestamp < %s::timestamp
                """, (ric_code, interval, start_date, end_date))
            else:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM lme_copper_intraday_data
                    WHERE ric_code = %s AND interval = %s
                """, (ric_code, interval))

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"データ件数取得エラー: {e}")
            return 0
        finally:
            cursor.close()

    def save_data(
        self,
        df: pd.DataFrame,
        ric_code: str,
        interval: str = '15min'
    ) -> int:
        """
        データをデータベースに保存（save_lme_intraday_dataのラッパー）

        Args:
            df: 保存するデータ
            ric_code: RICコード
            interval: データ間隔

        Returns:
            保存した行数
        """
        if not self.conn:
            self.connect()

        return self.save_lme_intraday_data(
            ric_code=ric_code,
            data=df,
            interval=interval
        )
