"""
Crystal Cluster UI
タブベースの設定管理インターフェース
"""

import streamlit as st
import sys
import os
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
backend_dir = current_dir.parent / 'backend'
sys.path.append(str(backend_dir))

from utils.config_manager import get_config_manager
from frontend.components.neo4j_config import render_neo4j_config
from frontend.components.api_config import render_api_config
from frontend.components.geode_tab import render_geode_tab
from frontend.components.advanced_config import render_advanced_config
from frontend.components.dictionary_manager import render_dictionary_manager

# ページ設定
st.set_page_config(
    page_title="Crystal Cluster",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定マネージャー
config_manager = get_config_manager()

def main():
    """メイン関数"""

    # ヘッダー
    st.title("💎 Crystal Cluster")
    st.markdown("*Knowledge Graph RAG System*")

    # タブ作成
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "基本設定",
        "AI設定",
        "Geode",
        "詳細設定",
        "辞書管理"
    ])

    # 基本設定タブ
    with tab1:
        render_neo4j_config(config_manager)

    # AI設定タブ
    with tab2:
        render_api_config(config_manager)

    # Geodeタブ
    with tab3:
        render_geode_tab(config_manager)

    # 詳細設定タブ
    with tab4:
        render_advanced_config(config_manager)

    # 辞書管理タブ
    with tab5:
        render_dictionary_manager(config_manager)

    # サイドバー
    with st.sidebar:
        st.header("システム情報")

        # 保存/読み込みボタン
        if st.button(" 設定を保存", type="primary"):
            if config_manager.save_config():
                st.success("設定を保存しました")
            else:
                st.error("保存に失敗しました")

        if st.button("🔄 設定をリロード"):
            # 設定を再読み込み
            config_manager = get_config_manager()
            st.success("設定をリロードしました")
            st.rerun()

        # デフォルト設定に戻す
        if st.button("🔙 デフォルトに戻す", type="secondary"):
            if st.confirm("本当にデフォルト設定に戻しますか？"):
                config_manager.reset_to_defaults()
                config_manager.save_config()
                st.success("デフォルト設定に戻しました")
                st.rerun()

        st.divider()

        # バージョン情報
        st.markdown("**バージョン:** 1.0.0")
        st.markdown("**Backend:** FastAPI")
        st.markdown("**Frontend:** Streamlit")

if __name__ == "__main__":
    main()