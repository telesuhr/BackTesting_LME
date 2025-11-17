# LME金属取引バックテストシステム

Refinitiv API経由でLME（ロンドン金属取引所）の6金属のデータを取得し、複数の取引戦略を一括でバックテストする統一システムです。

## 概要

このシステムは以下の機能を提供します：

- **6メタル対応**: 銅、アルミニウム、亜鉛、ニッケル、鉛、錫の統一管理
- **データ取得**: 1分足データ取得とPostgreSQLキャッシュ機能
- **4つの取引戦略**: ボリンジャーバンド、モメンタム、RSI逆張り、BB+RSI組み合わせ
- **一括バックテスト**: 全メタル×全戦略の組み合わせを1コマンドで実行
- **自動サマリー生成**: CSV/Markdown形式のパフォーマンスレポート自動作成
- **詳細可視化**: 4パネルチャート（価格・エクイティ・月別P&L・勝敗分布）

## プロジェクト構造

```
LMECopperTrading/
├── config/
│   └── metals_config.py          # 統一設定ファイル（6メタル×4戦略）
├── src/
│   ├── data/
│   │   ├── lme_client.py         # Refinitiv API クライアント
│   │   └── lme_db_manager.py     # PostgreSQL データベース管理
│   └── strategy/
│       ├── bollinger_bands.py    # ボリンジャーバンド戦略
│       ├── momentum.py           # モメンタム戦略（MA クロスオーバー）
│       ├── rsi_reversal.py       # RSI逆張り戦略
│       └── bollinger_rsi_combined.py # BB+RSI組み合わせ戦略
├── scripts/
│   ├── data_fetch/
│   │   └── fetch_all_metals.py   # 統一データ取得スクリプト
│   ├── backtest/
│   │   ├── run_all_backtests.py  # 統一バックテスト実行
│   │   └── generate_summary.py   # 結果サマリー自動生成
│   └── archive/                  # 旧スクリプト（参考用）
├── outputs/
│   ├── copper/                   # 銅の分析結果
│   ├── aluminium/                # アルミニウムの分析結果
│   ├── zinc/                     # 亜鉛の分析結果
│   ├── nickel/                   # ニッケルの分析結果
│   ├── lead/                     # 鉛の分析結果
│   ├── tin/                      # 錫の分析結果
│   └── summary/                  # パフォーマンスサマリー（CSV/MD）
├── database/
│   └── lme_schema.sql            # データベーススキーマ
└── config.json                   # API設定ファイル
```

## 対象商品

| 商品 | RICコード | 英名 | データ状況 |
|------|----------|------|-----------|
| 銅 | CMCU3 | Copper | ✅ 2024-11-11 ~ 2025-11-14 (1min) |
| アルミニウム | CMAL3 | Aluminium | ✅ 2024-11-11 ~ 2025-11-14 (1min) |
| 亜鉛 | CMZN3 | Zinc | ✅ 2024-11-11 ~ 2025-11-14 (1min) |
| ニッケル | CMNI3 | Nickel | 未取得 |
| 鉛 | CMPB3 | Lead | 未取得 |
| 錫 | CMSN3 | Tin | 未取得 |

## 取引戦略

### 1. ボリンジャーバンド戦略（平均回帰）
- **エントリー**: 2σバンドタッチで逆張り
- **イグジット**: エントリー価格から±2σ移動
- **パラメーター**: 期間20、2.0σ
- **ポジションサイズ**: 固定100MT

### 2. モメンタム戦略（トレンドフォロー）
- **エントリー**: MA5/20のゴールデン/デッドクロス
- **イグジット**: 逆クロス、または利食い2%・損切り1%
- **パラメーター**: 短期MA5、長期MA20
- **ポジションサイズ**: 固定100MT

### 3. RSI逆張り戦略（平均回帰）
- **エントリー**: RSI < 30（買い）/ RSI > 70（売り）
- **イグジット**: RSI中立帯（40-60）復帰
- **パラメーター**: RSI期間14、売買閾値30/70
- **ポジションサイズ**: 固定100MT

### 4. BB+RSI組み合わせ戦略（高精度平均回帰）
- **エントリー**: ボリンジャー2σタッチ AND RSI売買ゾーン（両方一致時のみ）
- **イグジット**: エントリー価格から±2σ移動
- **パラメーター**: BB期間20/2.0σ、RSI期間14（30/70）
- **ポジションサイズ**: 固定100MT

## 最新バックテスト結果（RSI逆張り戦略）

**期間**: 2024-11-11 ~ 2025-10-30
**データ**: 1分足 → 15分足リサンプリング
**初期資本**: $100,000

| メタル | リターン | 勝率 | Sharpe | Max DD | トレード数 | P&L |
|--------|----------|------|--------|--------|-----------|-----|
| 銅 | **121.2%** 🏆 | 67.2% | 6.36 | 94.5% | 920回 | $121.2k |
| 亜鉛 | **49.4%** | 65.8% | 7.58 | 15.9% | 876回 | $49.4k |
| アルミニウム | -6.3% | 66.1% | -0.29 | 29.2% | 875回 | $-6.3k |

**統計サマリー**:
- 平均リターン: 54.74%
- プラスリターン: 2/3銘柄 (66.7%)
- 平均勝率: 66.3%
- 平均Sharpe: 4.55

## セットアップ

### 1. データベースの準備

```bash
# データベースを作成
createdb -U postgres lme_copper_db

# スキーマを適用
psql -U postgres -d lme_copper_db -f database/lme_schema.sql
```

### 2. APIキーの設定

**方法1: config.jsonファイル（推奨）**

```bash
# テンプレートをコピー
cp config.json.template config.json

# config.jsonを編集してAPIキーを設定
# {
#   "refinitiv_api_key": "YOUR_API_KEY_HERE"
# }
```

**方法2: 環境変数**

```bash
export REFINITIV_API_KEY="your_api_key_here"
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主な依存パッケージ：
- refinitiv-data >= 1.0.0
- pandas >= 2.0.0
- psycopg2-binary >= 2.9.0
- matplotlib >= 3.7.0

## 使用方法

### 1. データ取得（統一スクリプト）

```bash
# 全6メタルのデータ取得（1min足）
python scripts/data_fetch/fetch_all_metals.py

# 特定メタルのみ取得
python scripts/data_fetch/fetch_all_metals.py --metals copper zinc nickel

# 期間指定
python scripts/data_fetch/fetch_all_metals.py --start 2024-01-01 --end 2025-01-01

# インターバル指定（デフォルト: 1min）
python scripts/data_fetch/fetch_all_metals.py --interval 1min

# 強制再取得（キャッシュ無視）
python scripts/data_fetch/fetch_all_metals.py --force-refresh
```

**重要**:
- 現在は**1分足データのみ**を取得します
- 5分足・15分足などは1分足からリサンプリングで生成できます
- Refinitiv APIは`15min`インターバルをサポートしていません
- サポート対象: `1min`, `5min`, `10min`, `30min`, `60min`, `hourly`, `daily`

### 2. バックテスト実行（統一スクリプト）

```bash
# 全メタル×全戦略のバックテストを一括実行
python scripts/backtest/run_all_backtests.py

# 特定メタルのみ実行
python scripts/backtest/run_all_backtests.py --metals copper zinc

# 特定戦略のみ実行
python scripts/backtest/run_all_backtests.py --strategies rsi bb_rsi

# 期間指定
python scripts/backtest/run_all_backtests.py --start 2024-11-11 --end 2025-11-11
```

**実行結果**:
- 各メタル×戦略の組み合わせごとに可視化PNG生成
- 自動的にCSV/Markdownサマリーが `outputs/summary/` に生成されます

### 3. 出力ファイル

**個別バックテスト結果**: `outputs/{metal}/`
- 例: `outputs/copper/rsi_backtest_1year_100mt_20251112_193418.png`

各グラフには以下が含まれます：
1. 価格チャート（エントリー/エグジットポイント表示）
2. エクイティカーブ（資本の推移と最大ドローダウン）
3. 月別損益（月ごとのP&L）
4. 勝敗分布（利益・損失のヒストグラム）

**パフォーマンスサマリー**: `outputs/summary/`
- CSV: `performance_summary_20251112_193433.csv`
- Markdown: `performance_summary_20251112_193433.md`

サマリーには以下が含まれます：
- 全体ランキング（リターン降順）
- メタル別パフォーマンス
- 戦略別パフォーマンス
- トップ10パフォーマー
- 統計サマリー

## Before/After（統一システムのメリット）

### Before（旧システム）
```bash
# 3メタル×4戦略 = 12回の個別実行が必要
python scripts/backtest/visualize_backtest_rsi_1year.py
python scripts/backtest/visualize_backtest_rsi_aluminium_1year.py
python scripts/backtest/visualize_backtest_rsi_zinc_1year.py
python scripts/backtest/visualize_backtest_bb_rsi_1year.py
# ... 合計12コマンド
```

### After（新システム）
```bash
# 1コマンドで全て完了
python scripts/backtest/run_all_backtests.py
```

**改善点**:
- スクリプト数: 31個 → 3個
- 実行コマンド: 12回 → 1回
- 設定管理: 分散 → 統一（config/metals_config.py）
- メタル追加: 各スクリプト修正 → config修正のみ
- 結果集計: 手動 → 自動（CSV/MD生成）

## データベース管理

### データ確認

```bash
# データベースに接続
psql -U postgres -d lme_copper_db

# データサマリーを表示
SELECT
    ric_code,
    interval,
    COUNT(*) as records,
    MIN(timestamp) as start_date,
    MAX(timestamp) as end_date
FROM lme_copper_intraday_data
GROUP BY ric_code, interval
ORDER BY ric_code, interval;

# 最新データを確認
SELECT * FROM lme_copper_intraday_data
WHERE ric_code = 'CMCU3' AND interval = '1min'
ORDER BY timestamp DESC
LIMIT 10;
```

### データベーススキーマ

```sql
CREATE TABLE lme_copper_intraday_data (
    id SERIAL PRIMARY KEY,
    ric_code VARCHAR(10) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ric_code, interval, timestamp)
);

CREATE INDEX idx_lme_intraday_ric_interval ON lme_copper_intraday_data(ric_code, interval);
CREATE INDEX idx_lme_intraday_timestamp ON lme_copper_intraday_data(timestamp);
```

## 設定ファイル（config/metals_config.py）

全メタルと全戦略の定義を一元管理：

```python
METALS_CONFIG = {
    'copper': {'ric': 'CMCU3', 'name': '銅', 'name_en': 'Copper'},
    'aluminium': {'ric': 'CMAL3', 'name': 'アルミニウム', 'name_en': 'Aluminium'},
    'zinc': {'ric': 'CMZN3', 'name': '亜鉛', 'name_en': 'Zinc'},
    'nickel': {'ric': 'CMNI3', 'name': 'ニッケル', 'name_en': 'Nickel'},
    'lead': {'ric': 'CMPB3', 'name': '鉛', 'name_en': 'Lead'},
    'tin': {'ric': 'CMSN3', 'name': '錫', 'name_en': 'Tin'}
}

STRATEGIES_CONFIG = {
    'bollinger': {...},
    'momentum': {...},
    'rsi': {...},
    'bb_rsi': {...}
}
```

新しいメタルや戦略を追加する場合は、この設定ファイルのみを編集します。

## トラブルシューティング

### エラー: "No default session created yet"

**原因**: Refinitiv API セッションが初期化されていない

**対処法**:
1. Refinitiv Workspace/Eikon Desktopアプリを起動
2. アプリ内でログイン完了を確認
3. スクリプト実行（自動的に`connect()`が呼ばれます）

### エラー: "Not supported interval value"

**原因**: Refinitiv APIがサポートしていないインターバルを指定

**対処法**:
- `15min`ではなく`1min`を取得
- 必要に応じてpandasの`resample()`で変換

サポート対象: `1min`, `5min`, `10min`, `30min`, `60min`, `hourly`, `daily`

### エラー: "データベース接続エラー"

**対処法**:
```bash
# PostgreSQLを起動
brew services start postgresql@14

# データベースの存在を確認
psql -U postgres -l

# 必要に応じてデータベース作成
createdb -U postgres lme_copper_db
```

## 技術スタック

- **データソース**: Refinitiv Data Platform API
- **データベース**: PostgreSQL 14+
- **言語**: Python 3.12
- **主要ライブラリ**:
  - `pandas` - データ処理・リサンプリング
  - `psycopg2` - PostgreSQL接続
  - `matplotlib` - グラフ描画
  - `refinitiv.data` - Refinitiv API クライアント

## ライセンス

Private use only.
