"""
辞書管理コンポーネント
"""

import streamlit as st
from utils.config_manager import get_config_manager


def render_dictionary_manager(config_manager: get_config_manager):
    """辞書管理UIを描画"""

    st.header("📚 辞書管理")

    st.info("辞書管理機能は現在開発中です。")

    # プレースホルダーコンテンツ
    st.subheader("機能予定")
    st.markdown("""
    - **同義語辞書**: 単語の同義語を管理
    - **専門用語辞書**: ドメイン固有の用語を定義
    - **ストップワード**: 除外する単語の管理
    - **カスタムルール**: テキスト処理のルールを定義
    """)

    # 現在の辞書設定表示（ダミー）
    st.subheader("現在の辞書設定")

    with st.expander("同義語辞書"):
        st.write("例: AI → 人工知能, 機械学習")
        st.write("例: NLP → 自然言語処理")

    with st.expander("専門用語辞書"):
        st.write("例: RAG → Retrieval-Augmented Generation")
        st.write("例: LLM → Large Language Model")

    with st.expander("ストップワード"):
        st.write("the, a, an, and, or, but, ...")

    # インポート/エクスポート
    st.subheader("辞書管理")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 辞書をインポート"):
            st.info("辞書ファイルを選択してください")

    with col2:
        if st.button("📤 辞書をエクスポート"):
            st.info("辞書ファイルをエクスポートします")

    # 設定表示
    with st.expander("辞書設定"):
        st.json({
            "synonyms": {},
            "terms": {},
            "stopwords": [],
            "rules": []
        })