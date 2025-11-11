# LME金属取引バックテストシステム

Refinitiv API経由でLME（ロンドン金属取引所）の銅・アルミニウム・亜鉛のデータを取得し、取引戦略のバックテストを実行するシステムです。

## 概要

このシステムは以下の機能を提供します：

- **データ取得**: LME銅（CMCU3）、アルミ（CMAL3）、亜鉛（CMZN3）の1分足データ取得
- **データ管理**: PostgreSQLデータベースへの自動保存（キャッシュ機能）
- **取引戦略**: ボリンジャーバンド（平均回帰）とモメンタム（トレンドフォロー）
- **バックテスト**: 1年間の履歴データを使った戦略検証
- **可視化**: 詳細なチャートとパフォーマンス統計の自動生成

## プロジェクト構造

```
LMECopperTrading/
├── src/                          # ソースコード
│   ├── data/                     # データ取得・管理
│   │   ├── lme_client.py        # Refinitiv API クライアント
│   │   └── lme_db_manager.py    # PostgreSQL データベース管理
│   └── strategy/                # 取引戦略
│       ├── bollinger_bands.py   # ボリンジャーバンド戦略
│       └── momentum.py          # モメンタム戦略（MA クロスオーバー）
├── scripts/                     # 実行スクリプト
│   ├── data_fetch/              # データ取得スクリプト
│   │   ├── fetch_*_data.py     # 各商品のデータ取得
│   │   └── resample_*.py       # データリサンプリング
│   └── backtest/                # バックテストスクリプト
│       ├── run_backtest_*.py   # バックテスト実行
│       └── visualize_*.py      # 結果可視化
├── outputs/                     # 出力ファイル
│   ├── copper/                  # 銅の分析結果
│   ├── aluminium/               # アルミニウムの分析結果
│   ├── zinc/                    # 亜鉛の分析結果
│   └── archive/                 # 古いファイル
└── config.json                  # API設定ファイル
```

## 対象商品

| 商品 | RICコード | 説明 |
|------|----------|------|
| 銅 | CMCU3 | LME銅3ヶ月先物 |
| アルミニウム | CMAL3 | LMEアルミニウム3ヶ月先物 |
| 亜鉛 | CMZN3 | LME亜鉛3ヶ月先物 |

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

## バックテスト結果サマリー

| 商品 | Bollinger Bands | Momentum |
|------|----------------|----------|
| **銅** | +27.02% (837回) | -601.89% (1,142回) |
| **アルミニウム** | -50.27% (873回) | -51.04% (1,082回) |
| **亜鉛** | **+85.65% (879回)** | -60.35% (1,063回) |

**期間**: 2024-11-11 ～ 2025-10-30
**データ**: 15分足OHLCV
**初期資本**: $100,000
**取引コスト**: ブローカー手数料 $0.5 + スプレッド 0.01%

### 主な発見

1. **亜鉛はボリンジャーバンド戦略に最適** - 85.65%のリターン、シャープレシオ7.91
2. **銅は中程度のパフォーマンス** - ボリンジャーで27%のリターン
3. **アルミニウムは両戦略で不調** - どちらもマイナスリターン
4. **モメンタム戦略は全商品で不良** - レンジ相場でのwhipsaw問題

## データベーススキーマ

```sql
CREATE TABLE lme_copper_intraday_data (
    id SERIAL PRIMARY KEY,
    ric_code VARCHAR(10),
    interval VARCHAR(10),
    timestamp TIMESTAMP,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ric_code, interval, timestamp)
);
```

## セットアップ

### 1. データベースの準備

PostgreSQLデータベース `lme_copper_db` が既に存在する場合：

```bash
# スキーマを適用
psql -U postgres -d lme_copper_db -f database/lme_schema.sql
```

新規作成する場合：

```bash
# データベースを作成
createdb -U postgres lme_copper_db

# スキーマを適用
psql -U postgres -d lme_copper_db -f database/lme_schema.sql
```

### 2. APIキーの設定

以下のいずれかの方法でRefinitiv APIキーを設定：

#### 方法1: config.jsonファイル（推奨）

```bash
# テンプレートをコピー
cp config.json.template config.json

# config.jsonを編集してAPIキーを設定
# {
#   "refinitiv_api_key": "YOUR_API_KEY_HERE",
#   ...
# }
```

#### 方法2: 環境変数

```bash
export REFINITIV_API_KEY="your_api_key_here"
# または
export EIKON_API_KEY="your_api_key_here"
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主な依存パッケージ：
- refinitiv-data >= 1.0.0
- pandas >= 2.0.0
- psycopg2-binary >= 2.9.0
- pytest >= 7.4.0

## 使用方法

### 1. データ取得

```bash
# 銅データ取得（2024-11-11 ~ 2025-11-11）
python scripts/data_fetch/fetch_missing_data.py

# アルミニウムデータ取得
python scripts/data_fetch/fetch_aluminium_data.py

# 亜鉛データ取得
python scripts/data_fetch/fetch_zinc_data.py
```

### 2. バックテスト実行

```bash
# 銅 - ボリンジャーバンド戦略
python scripts/backtest/visualize_backtest_1year.py

# 銅 - モメンタム戦略
python scripts/backtest/visualize_backtest_momentum_1year.py

# アルミニウム - ボリンジャーバンド戦略
python scripts/backtest/visualize_backtest_aluminium_1year.py

# アルミニウム - モメンタム戦略
python scripts/backtest/visualize_backtest_momentum_aluminium_1year.py

# 亜鉛 - ボリンジャーバンド戦略
python scripts/backtest/visualize_backtest_zinc_1year.py

# 亜鉛 - モメンタム戦略
python scripts/backtest/visualize_backtest_momentum_zinc_1year.py
```

### 3. 出力ファイル

バックテスト結果は `outputs/` フォルダに商品別に保存されます：

- `outputs/copper/` - 銅の分析結果
- `outputs/aluminium/` - アルミニウムの分析結果
- `outputs/zinc/` - 亜鉛の分析結果

各グラフには以下が含まれます：
1. 価格チャート（エントリー/エグジットポイント表示）
2. エクイティカーブ（資本の推移と最大ドローダウン）
3. 月別損益（月ごとのP&L）
4. 勝敗分布（利益・損失のヒストグラム）

ファイル名例: `bollinger_backtest_zinc_1year_100mt_20251111_185145.png`

## データベース確認

### psqlで直接確認

```bash
# データベースに接続
psql -U postgres -d lme_copper_db

# データサマリーを表示
SELECT * FROM lme_data_summary;

# 最新データを確認
SELECT * FROM lme_copper_intraday_data
WHERE ric_code = 'CMCU3'
ORDER BY timestamp DESC
LIMIT 10;

# レコード数を確認
SELECT ric_code, interval, COUNT(*) as count
FROM lme_copper_intraday_data
GROUP BY ric_code, interval;
```

## 技術スタック

- **データソース**: Refinitiv Data Platform API
- **データベース**: PostgreSQL
- **言語**: Python 3.12
- **主要ライブラリ**:
  - `pandas` - データ処理
  - `psycopg2` - PostgreSQL接続
  - `matplotlib` - グラフ描画
  - `refinitiv.data` - Refinitiv API

## トラブルシューティング

### エラー: "Refinitiv API接続失敗"

**原因**:
- Refinitiv Workspace/Eikon Desktopアプリが起動していない
- APIキーが正しく設定されていない

**対処法**:
1. Refinitiv Workspace を起動
2. APIキーを確認（config.json または環境変数）
3. ネットワーク接続を確認

### エラー: "データベース接続エラー"

**原因**: PostgreSQLが起動していない、または接続情報が間違っている

**対処法**:
```bash
# PostgreSQLの状態を確認
pg_ctl status

# PostgreSQLを起動
brew services start postgresql@14

# データベースの存在を確認
psql -U postgres -l
```

### エラー: "データが取得できませんでした"

**原因**:
- 指定した期間にデータが存在しない（土日祝日など）
- RICコードが無効または権限がない

**対処法**:
1. 営業日のデータを指定
2. RICコードを確認（CMCU3は最も流動性が高い）
3. Refinitivの権限を確認

## ライセンス

Private use only.
