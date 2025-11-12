"""
LMEメタル取引設定ファイル

6メタル（銅、アルミニウム、亜鉛、ニッケル、鉛、錫）の統一設定
"""
from typing import Dict, Any

# メタル定義（6メタル）
METALS_CONFIG: Dict[str, Dict[str, Any]] = {
    'copper': {
        'ric': 'CMCU3',
        'name': '銅',
        'name_en': 'Copper',
        'unit': 'MT',
        'description': 'LME銅3ヶ月先物'
    },
    'aluminium': {
        'ric': 'CMAL3',
        'name': 'アルミニウム',
        'name_en': 'Aluminium',
        'unit': 'MT',
        'description': 'LMEアルミニウム3ヶ月先物'
    },
    'zinc': {
        'ric': 'CMZN3',
        'name': '亜鉛',
        'name_en': 'Zinc',
        'unit': 'MT',
        'description': 'LME亜鉛3ヶ月先物'
    },
    'nickel': {
        'ric': 'CMNI3',
        'name': 'ニッケル',
        'name_en': 'Nickel',
        'unit': 'MT',
        'description': 'LMEニッケル3ヶ月先物'
    },
    'lead': {
        'ric': 'CMPB3',
        'name': '鉛',
        'name_en': 'Lead',
        'unit': 'MT',
        'description': 'LME鉛3ヶ月先物'
    },
    'tin': {
        'ric': 'CMSN3',
        'name': '錫',
        'name_en': 'Tin',
        'unit': 'MT',
        'description': 'LME錫3ヶ月先物'
    }
}

# 戦略定義
STRATEGIES_CONFIG: Dict[str, Dict[str, Any]] = {
    'bollinger': {
        'name': 'ボリンジャーバンド',
        'name_en': 'Bollinger Bands',
        'class_name': 'BollingerBandsStrategy',
        'module': 'src.strategy.bollinger_bands',
        'description': '平均回帰型（2σバンドタッチで逆張り）',
        'params': {
            'period': 20,
            'std': 2.0,
            'take_profit_pct': 0.015,
            'stop_loss_pct': 0.015,
            'max_positions': 1,
            'fixed_position_size': 100.0
        }
    },
    'momentum': {
        'name': 'モメンタム',
        'name_en': 'Momentum',
        'class_name': 'MomentumStrategy',
        'module': 'src.strategy.momentum',
        'description': 'トレンドフォロー型（MA5/20クロスオーバー）',
        'params': {
            'short_window': 5,
            'long_window': 20,
            'take_profit_pct': 0.02,
            'stop_loss_pct': 0.01,
            'max_positions': 1,
            'fixed_position_size': 100.0
        }
    },
    'rsi': {
        'name': 'RSI逆張り',
        'name_en': 'RSI Reversal',
        'class_name': 'RSIReversalStrategy',
        'module': 'src.strategy.rsi_reversal',
        'description': '平均回帰型（RSI 30/70で売買、40-60で決済）',
        'params': {
            'rsi_period': 14,
            'rsi_oversold': 30.0,
            'rsi_overbought': 70.0,
            'rsi_neutral_low': 40.0,
            'rsi_neutral_high': 60.0,
            'max_positions': 1,
            'fixed_position_size': 100.0
        }
    },
    'bb_rsi': {
        'name': 'BB+RSI組み合わせ',
        'name_en': 'BB+RSI Combined',
        'class_name': 'BollingerRSICombinedStrategy',
        'module': 'src.strategy.bollinger_rsi_combined',
        'description': '高精度平均回帰型（BB 2σ AND RSI売買ゾーン）',
        'params': {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30.0,
            'rsi_overbought': 70.0,
            'max_positions': 1,
            'fixed_position_size': 100.0
        }
    }
}

# バックテスト共通設定
BACKTEST_CONFIG: Dict[str, Any] = {
    'initial_capital': 100000.0,
    'risk_per_trade': 0.02,
    'broker_commission_usd': 0.5,
    'spread_pct': 0.0001,
    'start_date': '2024-11-11',
    'end_date': '2025-11-11',
    'interval': '15min'
}

# データベース設定
DATABASE_CONFIG: Dict[str, Any] = {
    'host': 'localhost',
    'port': '5432',
    'database': 'lme_copper_db',
    'user': 'postgres',
    'password': ''
}

# 出力設定
OUTPUT_CONFIG: Dict[str, Any] = {
    'base_dir': 'outputs',
    'summary_dir': 'outputs/summary',
    'graph_dpi': 150,
    'graph_format': 'png'
}


def get_metal_config(metal_key: str) -> Dict[str, Any]:
    """メタル設定を取得"""
    if metal_key not in METALS_CONFIG:
        raise ValueError(f"Unknown metal: {metal_key}. Available: {list(METALS_CONFIG.keys())}")
    return METALS_CONFIG[metal_key]


def get_strategy_config(strategy_key: str) -> Dict[str, Any]:
    """戦略設定を取得"""
    if strategy_key not in STRATEGIES_CONFIG:
        raise ValueError(f"Unknown strategy: {strategy_key}. Available: {list(STRATEGIES_CONFIG.keys())}")
    return STRATEGIES_CONFIG[strategy_key]


def get_all_metals() -> list:
    """全メタルキーを取得"""
    return list(METALS_CONFIG.keys())


def get_all_strategies() -> list:
    """全戦略キーを取得"""
    return list(STRATEGIES_CONFIG.keys())
