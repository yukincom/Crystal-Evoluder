#　エキスパートモード
from .presets import RECOMMENDED_PRESETS, PARAMETER_DOCS

def load_config(mode: str, user_values: dict | None = None):
    """
    設定をロード
    
    Args:
        mode: "preset" または "expert"
        user_values: ユーザー指定の値
    
    Returns:
        完全な設定辞書
    """
    if mode == "preset":
        preset_name = user_values.get("preset_name", "general")
        
        if preset_name not in RECOMMENDED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {list(RECOMMENDED_PRESETS.keys())}"
            )
        
        config = RECOMMENDED_PRESETS[preset_name].copy()
        
        # LLM設定は別途指定
        config['llm_model'] = user_values.get('llm_model', 'gpt-4o-mini')
        config['llm_timeout'] = user_values.get('llm_timeout', 120.0)
        
        return config
    
    elif mode == "expert":
        # 必須パラメータチェック
        required = [
            "entity_linking_threshold",
            "retrieval_chunk_size",
            "retrieval_chunk_overlap",
            "graph_chunk_size",
            "graph_chunk_overlap",
            "relation_compat_threshold",
            "final_weight_cutoff",
            "max_triplets_per_chunk"
        ]
        
        missing = [r for r in required if r not in user_values]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # 範囲チェック（警告のみ）
        warnings = []
        for key, value in user_values.items():
            if key in PARAMETER_DOCS:
                min_val, max_val = PARAMETER_DOCS[key]['range']
                if not (min_val <= value <= max_val):
                    warnings.append(
                        f"{key}={value} is outside recommended range "
                        f"[{min_val}, {max_val}]"
                    )
        
        config = user_values.copy()
        config['warnings'] = warnings
        
        return config
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'preset' or 'expert'")


def get_parameter_info(param_name: str = None) -> dict:
    """
    パラメータの説明を取得（UI用）
    
    Args:
        param_name: 特定のパラメータ名（Noneなら全て）
    
    Returns:
        パラメータ情報の辞書
    """
    if param_name:
        return PARAMETER_DOCS.get(param_name, {})
    return PARAMETER_DOCS
