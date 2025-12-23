"""
詳細設定コンポーネント
"""

import streamlit as st
from utils.config_manager import get_config_manager

def render_advanced_config(config_manager: get_config_manager):
    """詳細設定UIを描画"""

    st.header("⚙️ 詳細設定")

    # パラメータ設定
    st.subheader("基本パラメータ")

    parameters = config_manager.get('parameters')

    col1, col2 = st.columns(2)

    with col1:
        entity_threshold = st.slider(
            "エンティティ統合閾値",
            min_value=0.80,
            max_value=0.95,
            value=parameters.get('entity_linking_threshold', 0.88),
            step=0.01,
            help="エンティティ統合の類似度閾値"
        )

        retrieval_size = st.slider(
            "Retrievalチャンクサイズ",
            min_value=256,
            max_value=768,
            value=parameters.get('retrieval_chunk_size', 320),
            step=32,
            help="Retrieval用チャンクサイズ"
        )

        graph_size = st.slider(
            "Graphチャンクサイズ",
            min_value=384,
            max_value=640,
            value=parameters.get('graph_chunk_size', 512),
            step=32,
            help="Graph用チャンクサイズ"
        )

    with col2:
        retrieval_overlap = st.slider(
            "Retrieval重複部分",
            min_value=50,
            max_value=200,
            value=parameters.get('retrieval_chunk_overlap', 120),
            step=10,
            help="Retrievalチャンクの重複部分"
        )

        graph_overlap = st.slider(
            "Graph重複部分",
            min_value=30,
            max_value=100,
            value=parameters.get('graph_chunk_overlap', 50),
            step=5,
            help="Graphチャンクの重複部分"
        )

        relation_threshold = st.slider(
            "関係相性閾値",
            min_value=0.05,
            max_value=0.15,
            value=parameters.get('relation_compat_threshold', 0.11),
            step=0.01,
            help="関係の相性判定閾値"
        )

    # Self-RAG設定
    st.subheader("Self-RAG設定")

    self_rag = config_manager.get('self_rag')

    enable_self_rag = st.checkbox(
        "Self-RAGを有効化",
        value=self_rag.get('enable', True),
        help="Self-RAG機能を有効にする"
    )

    if enable_self_rag:
        col1, col2 = st.columns(2)

        with col1:
            confidence_threshold = st.slider(
                "信頼性閾値",
                min_value=0.0,
                max_value=1.0,
                value=self_rag.get('confidence_threshold', 0.75),
                step=0.05,
                help="回答の信頼性閾値"
            )

            max_retries = st.slider(
                "最大リトライ回数",
                min_value=0,
                max_value=5,
                value=self_rag.get('max_retries', 1),
                help="Self-RAGの最大リトライ回数"
            )

        with col2:
            token_budget = st.slider(
                "トークン予算",
                min_value=10000,
                max_value=200000,
                value=self_rag.get('token_budget', 100000),
                step=10000,
                help="Self-RAGのトークン予算"
            )

    # 処理設定
    st.subheader("処理設定")

    processing = config_manager.get('processing')

    col1, col2 = st.columns(2)

    with col1:
        enable_duplicate_check = st.checkbox(
            "重複チェック有効",
            value=processing.get('enable_duplicate_check', True),
            help="ファイルの重複チェックを行う"
        )

        enable_provenance = st.checkbox(
            "出典追跡有効",
            value=processing.get('enable_provenance', True),
            help="データの出典を追跡する"
        )

    with col2:
        max_workers = st.slider(
            "最大ワーカー数",
            min_value=1,
            max_value=8,
            value=processing.get('max_workers', 4),
            help="並列処理のワーカー数"
        )

    # 更新ボタン
    if st.button("🔄 詳細設定を更新", type="primary"):
        # パラメータ更新
        config_manager.set('parameters', 'entity_linking_threshold', entity_threshold)
        config_manager.set('parameters', 'retrieval_chunk_size', retrieval_size)
        config_manager.set('parameters', 'retrieval_chunk_overlap', retrieval_overlap)
        config_manager.set('parameters', 'graph_chunk_size', graph_size)
        config_manager.set('parameters', 'graph_chunk_overlap', graph_overlap)
        config_manager.set('parameters', 'relation_compat_threshold', relation_threshold)

        # Self-RAG更新
        config_manager.set('self_rag', 'enable', enable_self_rag)
        if enable_self_rag:
            config_manager.set('self_rag', 'confidence_threshold', confidence_threshold)
            config_manager.set('self_rag', 'max_retries', max_retries)
            config_manager.set('self_rag', 'token_budget', token_budget)

        # 処理設定更新
        config_manager.set('processing', 'enable_duplicate_check', enable_duplicate_check)
        config_manager.set('processing', 'enable_provenance', enable_provenance)
        config_manager.set('processing', 'max_workers', max_workers)

        st.success("詳細設定を更新しました")

    # 現在の設定表示
    with st.expander("現在の詳細設定"):
        st.json({
            'parameters': parameters,
            'self_rag': self_rag,
            'processing': processing
        })