"""
Geodeタブコンポーネント
"""

import streamlit as st
import os
from pathlib import Path
from utils.config_manager import get_config_manager

def render_geode_tab(config_manager: get_config_manager):
    """GeodeタブUIを描画"""

    st.header("🏔️ Geode")

    # Geode設定を取得
    geode_config = config_manager.get('geode')

    # 入力ディレクトリ設定
    st.subheader("入力設定")
    input_dir = st.text_input(
        "入力ディレクトリ",
        value=geode_config.get('input_dir', ''),
        help="処理対象のファイルがあるディレクトリ"
    )

    # ディレクトリ選択ボタン
    if st.button("📁 入力ディレクトリを選択"):
        # ブラウザではディレクトリ選択が難しいので、テキスト入力のみ
        st.info("パスを直接入力してください")

    # 出力ディレクトリ設定
    output_dir = st.text_input(
        "出力ディレクトリ",
        value=geode_config.get('output_dir', './output'),
        help="処理結果を出力するディレクトリ"
    )

    # パターン設定
    patterns = geode_config.get('patterns', ['*.pdf', '*.md', '*.docx'])
    patterns_str = ', '.join(patterns)
    new_patterns = st.text_input(
        "対象ファイルパターン",
        value=patterns_str,
        help="処理対象のファイルパターン（カンマ区切り）"
    )

    # 更新ボタン
    if st.button("🔄 Geode設定を更新", type="primary"):
        config_manager.set('geode', 'input_dir', input_dir)
        config_manager.set('geode', 'output_dir', output_dir)
        config_manager.set('geode', 'patterns', [p.strip() for p in new_patterns.split(',')])
        st.success("Geode設定を更新しました")

    st.divider()

    # ファイル処理実行
    st.subheader("ファイル処理")

    if st.button("🚀 処理実行", type="primary"):
        with st.spinner("ファイルを処理中..."):
            # TODO: Geode処理の実装
            import time
            time.sleep(2)  # ダミー処理

            # 処理結果の表示
            st.success("✅ 処理完了")
            st.info("処理されたファイル数: 5")
            st.info("生成されたノード数: 150")
            st.info("生成された関係数: 200")

    # 処理履歴
    st.subheader("処理履歴")
    with st.expander("最近の処理"):
        st.write("2024-01-15 10:30: 論文PDF 3ファイルを処理")
        st.write("2024-01-14 15:20: マニュアル文書 2ファイルを処理")
        st.write("2024-01-13 09:15: 技術資料 1ファイルを処理")

    # 現在の設定表示
    with st.expander("現在のGeode設定"):
        st.json(geode_config)