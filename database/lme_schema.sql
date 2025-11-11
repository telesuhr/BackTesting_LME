-- LME Copperティックデータ（1分足）保存テーブル
CREATE TABLE IF NOT EXISTS lme_copper_intraday_data (
    id SERIAL PRIMARY KEY,
    ric_code VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    interval VARCHAR(10) DEFAULT '1min',
    currency VARCHAR(3) DEFAULT 'USD',
    unit VARCHAR(10) DEFAULT 'MT',  -- Metric Ton
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ric_code, timestamp, interval)
);

-- 検索高速化のためのインデックス
CREATE INDEX IF NOT EXISTS idx_lme_ric_timestamp ON lme_copper_intraday_data(ric_code, timestamp);
CREATE INDEX IF NOT EXISTS idx_lme_timestamp ON lme_copper_intraday_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_lme_ric ON lme_copper_intraday_data(ric_code);

-- LMEデータ取得ログテーブル（デバッグ用）
CREATE TABLE IF NOT EXISTS lme_data_fetch_log (
    id SERIAL PRIMARY KEY,
    ric_code VARCHAR(20) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    interval VARCHAR(10),
    source VARCHAR(20), -- 'api' or 'cache'
    records_count INT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_lme_fetch_log_ric ON lme_data_fetch_log(ric_code, fetched_at);

-- LME RICコードマスタテーブル
CREATE TABLE IF NOT EXISTS lme_ric_master (
    id SERIAL PRIMARY KEY,
    ric_code VARCHAR(20) NOT NULL UNIQUE,
    metal_name VARCHAR(50) NOT NULL,
    tenor_type VARCHAR(50),  -- '3M', '15M', 'Cash' etc.
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LME Copper RICコードの初期データ挿入
INSERT INTO lme_ric_master (ric_code, metal_name, tenor_type, description) VALUES
    ('CMCU3', 'Copper', '3M', 'LME Copper 3-month forward'),
    ('CMCU15', 'Copper', '15M', 'LME Copper 15-month forward'),
    ('CMCU0', 'Copper', 'Cash', 'LME Copper Electronic Settlement'),
    ('CMCUc1', 'Copper', '1st Month', 'LME Copper 1st Month Generic'),
    ('CMCUc2', 'Copper', '2nd Month', 'LME Copper 2nd Month Generic'),
    ('CMCUc3', 'Copper', '3rd Month', 'LME Copper 3rd Month Generic'),
    ('CMCUc6', 'Copper', '6th Month', 'LME Copper 6th Month Generic'),
    ('CMCUc12', 'Copper', '12th Month', 'LME Copper 12th Month Generic')
ON CONFLICT (ric_code) DO NOTHING;

-- データベース情報確認用ビュー
CREATE OR REPLACE VIEW lme_data_summary AS
SELECT
    ric_code,
    interval,
    MIN(timestamp) as earliest_data,
    MAX(timestamp) as latest_data,
    COUNT(*) as total_records,
    COUNT(DISTINCT DATE(timestamp)) as trading_days
FROM lme_copper_intraday_data
GROUP BY ric_code, interval
ORDER BY ric_code, interval;
