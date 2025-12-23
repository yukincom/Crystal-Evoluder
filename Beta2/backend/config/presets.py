# config/presets.py

PARAMETER_DOCS = {
    'entity_linking_threshold': {
        'description': 'エンティティ統合の類似度閾値',
        'range': (0.80, 0.95),
        'recommended': 0.88,
        'notes': '''
        - 高い（0.92~0.95）: 固有名詞が多い分野（Bio、固有技術名）
        - 中間（0.88~0.90）: 汎用・バランス型
        - 低い（0.85~0.87）: 同義語が多い分野（NLP）
        '''
    },
    'retrieval_chunk_size': {
        'description': 'Retrieval用チャンクサイズ（文字数）',
        'range': (256, 768),
        'recommended': 320,
        'notes': '''
        - 小さい（256~320）: 精度重視、情報密度を下げる
        - 大きい（512~768）: 文脈重視、長い説明が多い文書
        '''
    },
    'retrieval_chunk_overlap': {
        'description': 'Retrievalチャンクの重複部分',
        'range': (50, 200),
        'recommended': 120,
        'notes': 'chunk_sizeの30~40%が目安'
    },
    'graph_chunk_size': {
        'description': 'Graph用チャンクサイズ（トリプレット抽出）',
        'range': (384, 640),
        'recommended': 512,
        'notes': 'トリプレット抽出の精度に影響。512が最適'
    },
    'graph_chunk_overlap': {
        'description': 'Graphチャンクの重複部分',
        'range': (30, 100),
        'recommended': 50,
        'notes': 'chunk_sizeの10%程度'
    },
    'relation_compat_threshold': {
        'description': '関係の相性判定閾値（Inter-document）',
        'range': (0.05, 0.15),
        'recommended': 0.11,
        'notes': '''
        - 厳しい（0.12~0.15）: 手法相関が強い分野（CV、ML）
        - 緩い（0.08~0.10）: 因果関係が複雑な分野（Bio、理論系）
        '''
    },
    'final_weight_cutoff': {
        'description': '最終エッジ重みのカットオフ',
        'range': (0.01, 0.05),
        'recommended': 0.035,
        'notes': '''
        - 高い（0.04~0.05）: 強い関係のみ残す（CV、ML）
        - 低い（0.02~0.03）: 弱い依存も残す（NLP、Graph）
        '''
    },
    'max_triplets_per_chunk': {
        'description': 'チャンクあたりの最大トリプレット数',
        'range': (8, 20),
        'recommended': 15,
        'notes': '''
        - 低い（8~12）: ノイズ削減、質重視
        - 高い（15~20）: 網羅性重視、質と量のバランス
        '''
    }
}

# 推奨プリセット（参考値として提示）
RECOMMENDED_PRESETS = {
    'general': {
        'entity_linking_threshold': 0.88,
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.11,
        'final_weight_cutoff': 0.035,
        'max_triplets_per_chunk': 15,
    },
    'strict_quality': {  # 質重視
        'entity_linking_threshold': 0.92,
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.13,
        'final_weight_cutoff': 0.04,
        'max_triplets_per_chunk': 12,
    },
    'high_coverage': {  # 網羅性重視
        'entity_linking_threshold': 0.85,
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.09,
        'final_weight_cutoff': 0.025,
        'max_triplets_per_chunk': 18,
    },
   'general': {
        'entity_linking_threshold': 0.88,
        'retrieval_chunk_size': 320,     
        'retrieval_chunk_overlap': 120,   
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.11,  
        'final_weight_cutoff': 0.035, 
        'max_triplets_per_chunk': 15,
    },
    'cv': {  # Computer Vision (CVPR, ICCV系)
        'entity_linking_threshold': 0.90,  # やや厳しめ（固有名詞多い）
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.13,  # ← 厳しめ（手法名が頻出）
        'final_weight_cutoff': 0.04,        # ← 高め（弱依存を削る）
        'max_triplets_per_chunk': 15,
    },
            'nlp': {  # NLP (ACL, EMNLP系)
        'entity_linking_threshold': 0.87,  # 緩め（同義語多い）
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.10,  # ← やや緩め
        'final_weight_cutoff': 0.03,
        'max_triplets_per_chunk': 15,
    },
    'ml': {  # Machine Learning (NeurIPS, ICML系)
        'entity_linking_threshold': 0.88,
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.12,  # ← 厳しめ（手法相関強い）
        'final_weight_cutoff': 0.04,
        'max_triplets_per_chunk': 15,
    },
    'graph': {  # Graph/Network系
        'entity_linking_threshold': 0.89,
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.11,
        'final_weight_cutoff': 0.035,
        'max_triplets_per_chunk': 15,
    },
    'bio': {  # Bioinformatics
        'entity_linking_threshold': 0.92,  # 厳しめ（固有名詞厳密）
        'retrieval_chunk_size': 320,
        'retrieval_chunk_overlap': 120,
        'graph_chunk_size': 512,
        'graph_chunk_overlap': 50,
        'relation_compat_threshold': 0.09,  # 緩め（因果関係複雑）
        'final_weight_cutoff': 0.03,
        'max_triplets_per_chunk': 15,
    }
}

def get_preset(preset_name: str = 'general') -> dict:
    """
    プリセット設定を取得

    Args:
        preset_name: プリセット名 ('general', 'cv', 'nlp', 'ml', 'graph', 'bio', 'strict_quality', 'high_coverage')

    Returns:
        設定辞書

    Raises:
        ValueError: 無効なプリセット名の場合
    """
    if preset_name not in RECOMMENDED_PRESETS:
        available = list(RECOMMENDED_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return RECOMMENDED_PRESETS[preset_name].copy()