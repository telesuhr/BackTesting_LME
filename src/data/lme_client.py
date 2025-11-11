"""
LME Copper専用 Refinitiv API接続モジュール

Refinitiv Data Platform APIを使用してLME Copperデータを取得
PostgreSQLキャッシュ機能を実装
"""
import refinitiv.data as rd
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging
import time
import json
import os
from .lme_db_manager import LMEDatabaseManager

logger = logging.getLogger(__name__)


class LMEDataClient:
    """LME Copper専用 Refinitiv API クライアント（DBキャッシュ機能付き）"""

    def __init__(self, app_key: str = None, use_cache: bool = True, db_config: dict = None):
        """
        Args:
            app_key: Refinitiv API キー（Noneの場合は設定ファイルから読み込み）
            use_cache: データベースキャッシュを使用するか
            db_config: データベース接続設定（Noneの場合は環境変数から読み込み）
        """
        # APIキーの設定
        if app_key is None:
            # config.jsonから読み込み
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../../config.json'
            )
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    app_key = config.get('refinitiv_api_key', config.get('eikon_api_key'))
            else:
                # 環境変数から読み込み
                app_key = os.getenv('REFINITIV_API_KEY', os.getenv('EIKON_API_KEY'))

        self.app_key = app_key
        self._session = None
        self.use_cache = use_cache
        self.db_manager = None

        if use_cache:
            try:
                self.db_manager = LMEDatabaseManager(db_config)
                self.db_manager.connect()
                logger.info("データベースキャッシュ機能を有効化")
            except Exception as e:
                logger.warning(f"データベース接続失敗、キャッシュ無効化: {e}")
                self.use_cache = False

    def connect(self):
        """APIセッションを開始"""
        try:
            session = rd.open_session(
                name='desktop.workspace',
                app_key=self.app_key
            )
            self._session = session
            logger.info("Refinitiv API接続成功")
        except Exception as e:
            logger.error(f"Refinitiv API接続失敗: {e}")
            raise

    def disconnect(self):
        """APIセッションを終了"""
        try:
            rd.close_session()
            self._session = None
            logger.info("Refinitiv API切断完了")
        except Exception as e:
            logger.error(f"Refinitiv API切断失敗: {e}")

        # データベース接続も切断
        if self.db_manager:
            self.db_manager.disconnect()

    def get_lme_intraday_data(
        self,
        ric_code: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """
        LME Copper分足データを取得（DBキャッシュ優先）

        Args:
            ric_code: RICコード（例: 'CMCU3'）
            start_date: 開始日時
            end_date: 終了日時
            interval: 時間間隔（'1min', '5min', '10min', '30min', '1h'等）

        Returns:
            OHLCV データフレーム
        """
        # 1. DBキャッシュから取得を試みる
        if self.use_cache and self.db_manager:
            cached_data = self.db_manager.get_lme_intraday_data(
                ric_code=ric_code,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )

            if cached_data is not None and not cached_data.empty:
                logger.info(f"{ric_code}: DBキャッシュから{len(cached_data)}行を取得 ✓")
                # ログに記録
                self.db_manager.log_lme_fetch(
                    ric_code=ric_code,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    source='cache',
                    records_count=len(cached_data)
                )
                return cached_data

        # 2. キャッシュにない場合、APIから取得
        logger.info(f"{ric_code}: DBキャッシュにデータなし、APIから取得...")

        try:
            # Refinitiv Data Platform APIを使用して分足データを取得
            data = rd.get_history(
                universe=ric_code,
                start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                end=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
                interval=interval
            )

            if data is None or data.empty:
                logger.warning(f"{ric_code} のデータが取得できませんでした")
                return None

            # データフレームの整形
            # カラム名のマッピング
            column_mapping = {
                'HIGH_1': 'high',
                'LOW_1': 'low',
                'OPEN_PRC': 'open',
                'TRDPRC_1': 'close',
                'ACVOL_UNS': 'volume'
            }

            # 存在するカラムのみマッピング
            existing_mapping = {k: v for k, v in column_mapping.items() if k in data.columns}
            data = data.rename(columns=existing_mapping)

            # 必要なカラムのみ抽出
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in data.columns]

            # volumeがある場合は追加
            if 'volume' in data.columns:
                available_cols.append('volume')

            data = data[available_cols]

            logger.info(
                f"{ric_code}: APIから{len(data)}行を取得 "
                f"({start_date.date()} - {end_date.date()})"
            )

            # 3. 取得したデータをDBに保存
            if self.use_cache and self.db_manager and not data.empty:
                saved_count = self.db_manager.save_lme_intraday_data(
                    ric_code=ric_code,
                    data=data,
                    interval=interval
                )
                logger.info(f"{ric_code}: {saved_count}行をDBに保存 ✓")

                # ログに記録
                self.db_manager.log_lme_fetch(
                    ric_code=ric_code,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    source='api',
                    records_count=len(data)
                )

            return data

        except Exception as e:
            logger.error(f"{ric_code} のデータ取得エラー: {e}")
            return None

    def get_lme_daily_data(
        self,
        ric_codes: list,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        LME日足データを一括取得

        Args:
            ric_codes: RICコードのリスト
            start_date: 開始日
            end_date: 終了日

        Returns:
            {ric_code: DataFrame} の辞書
        """
        results = {}

        for ric_code in ric_codes:
            try:
                # 日足データを取得
                data = rd.get_history(
                    universe=ric_code,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='daily'
                )

                if data is not None and not data.empty:
                    # カラム名を小文字に変換
                    data.columns = [col.lower() for col in data.columns]
                    results[ric_code] = data
                    logger.info(f"{ric_code}: {len(data)}行の日足データを取得")
                else:
                    logger.warning(f"{ric_code} のデータが取得できませんでした")

                # API制限対策: レート制限を考慮
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"{ric_code} のデータ取得エラー: {e}")
                continue

        return results
