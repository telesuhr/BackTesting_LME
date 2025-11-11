"""
LME Copperデータ取得のテストコード

TDD原則：このテストを先に作成し、失敗を確認してから実装を進める
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLMECopperDataRetrieval:
    """LME Copperデータ取得テスト"""

    def test_get_lme_copper_1min_data(self):
        """
        期待される動作：
        - CMCU3（LME Copper 3M）の1分足データを取得できる
        - データにはopen, high, low, close, volumeカラムが含まれる
        - タイムスタンプがDatetimeIndexとして設定されている
        - データが空でない
        """
        # このテストは最初は失敗するはず（実装がまだないため）
        from src.data.lme_client import LMEDataClient

        # テスト用の日付範囲（直近3日間）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        # LMEデータクライアントを初期化
        client = LMEDataClient()
        client.connect()

        try:
            # 1分足データを取得
            data = client.get_lme_intraday_data(
                ric_code='CMCU3',
                start_date=start_date,
                end_date=end_date,
                interval='1min'
            )

            # 検証1: データが返されること
            assert data is not None, "データが取得できませんでした"
            assert not data.empty, "データが空です"

            # 検証2: 必要なカラムが存在すること
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                assert col in data.columns, f"{col}カラムが存在しません"

            # 検証3: インデックスがDatetimeIndexであること
            assert isinstance(data.index, pd.DatetimeIndex), \
                "インデックスがDatetimeIndexではありません"

            # 検証4: データの整合性チェック
            # high >= low, high >= open, high >= close
            assert (data['high'] >= data['low']).all(), \
                "high < lowのデータが存在します"
            assert (data['high'] >= data['open']).all(), \
                "high < openのデータが存在します"
            assert (data['high'] >= data['close']).all(), \
                "high < closeのデータが存在します"

            logger.info(f"✓ テスト成功: {len(data)}行のデータを取得")
            logger.info(f"  期間: {data.index.min()} - {data.index.max()}")
            logger.info(f"  カラム: {list(data.columns)}")

        finally:
            client.disconnect()

    def test_save_lme_data_to_database(self):
        """
        期待される動作：
        - 取得したLME Copperデータをデータベースに保存できる
        - 保存したデータを再度取得できる
        - データの整合性が保たれている
        """
        from src.data.lme_client import LMEDataClient

        # テスト用の日付範囲（直近1日）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        client = LMEDataClient(use_cache=True)
        client.connect()

        try:
            # データを取得
            original_data = client.get_lme_intraday_data(
                ric_code='CMCU3',
                start_date=start_date,
                end_date=end_date,
                interval='1min'
            )

            assert original_data is not None, "データが取得できませんでした"

            # データベースに保存（内部で自動的に保存される）
            # 再度同じデータを取得（キャッシュから取得されるはず）
            cached_data = client.get_lme_intraday_data(
                ric_code='CMCU3',
                start_date=start_date,
                end_date=end_date,
                interval='1min'
            )

            # 検証: キャッシュから取得したデータが元のデータと一致すること
            assert cached_data is not None, "キャッシュデータが取得できませんでした"
            assert len(cached_data) == len(original_data), \
                "キャッシュデータの行数が一致しません"

            logger.info(f"✓ テスト成功: {len(cached_data)}行のデータを保存・取得")

        finally:
            client.disconnect()

    def test_multiple_ric_codes(self):
        """
        期待される動作：
        - 複数のRICコード（CMCU3, CMCU15など）のデータを取得できる
        - 各RICコードで適切なデータが返される
        """
        from src.data.lme_client import LMEDataClient

        ric_codes = ['CMCU3', 'CMCU15']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)

        client = LMEDataClient()
        client.connect()

        try:
            results = {}
            for ric_code in ric_codes:
                data = client.get_lme_intraday_data(
                    ric_code=ric_code,
                    start_date=start_date,
                    end_date=end_date,
                    interval='1min'
                )
                results[ric_code] = data

                # 各RICコードでデータが取得できることを確認
                assert data is not None, f"{ric_code}のデータが取得できませんでした"
                logger.info(f"✓ {ric_code}: {len(data)}行のデータを取得")

            # すべてのRICコードでデータが取得できたことを確認
            assert len(results) == len(ric_codes), \
                "一部のRICコードでデータが取得できませんでした"

        finally:
            client.disconnect()


if __name__ == '__main__':
    # テストを実行
    pytest.main([__file__, '-v', '-s'])
