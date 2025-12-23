"""
Neo4j設定コンポーネント
"""

import streamlit as st
from ...utils.config_manager import get_config_manager
from ...utils.validators import validate_neo4j_connection


def render_neo4j_config():
    """Neo4j設定UIを描画"""

    st.subheader("🗄️ Neo4j データベース設定")

    config_mgr = get_config_manager()
    neo4j_config = config_mgr.get_neo4j_config()

    # 接続情報入力
    col1, col2 = st.columns([3, 1])

    with col1:
        url = st.text_input(
            "接続URL",
            value=neo4j_config.get('url', 'bolt://localhost:7687'),
            help="Neo4jサーバーのURL（例: bolt://localhost:7687）",
            key="neo4j_url"
        )

    with col2:
        database = st.text_input(
            "データベース",
            value=neo4j_config.get('database', 'neo4j'),
            help="使用するデータベース名",
            key="neo4j_database"
        )

    col3, col4 = st.columns(2)

    with col3:
        username = st.text_input(
            "ユーザー名",
            value=neo4j_config.get('username', 'neo4j'),
            key="neo4j_username"
        )

    with col4:
        password = st.text_input(
            "パスワード",
            value=neo4j_config.get('password', ''),
            type="password",
            help="Neo4jのパスワード",
            key="neo4j_password"
        )

    # 接続テストと保存
    col5, col6, col7 = st.columns([2, 2, 1])

    with col5:
        if st.button("🔌 接続テスト", use_container_width=True):
            with st.spinner("接続確認中..."):
                success, error_msg = validate_neo4j_connection(
                    url, username, password, database
                )

                if success:
                    st.success("✅ 接続成功！")
                    st.session_state['neo4j_connected'] = True
                else:
                    st.error(f"❌ {error_msg}")
                    st.session_state['neo4j_connected'] = False

    with col6:
        if st.button("💾 設定を保存", use_container_width=True):
            config_mgr.set_neo4j_config(url, username, password, database)

            if config_mgr.save_config():
                st.success("✅ 設定を保存しました")
            else:
                st.error("❌ 保存に失敗しました")

    with col7:
        if st.button("🔄", help="デフォルトに戻す"):
            st.rerun()

    # 接続状態の表示
    if 'neo4j_connected' in st.session_state:
        if st.session_state['neo4j_connected']:
            st.info("🟢 Neo4jに接続されています")
        else:
            st.warning("🔴 Neo4jに接続されていません")

    # 詳細情報（折りたたみ）
    with st.expander("ℹ️ Neo4j設定のヘルプ"):
        st.markdown("""
        ### セットアップ手順

        1. **Neo4jをインストール**
           ```bash
           # Dockerで起動する場合
           docker run -d \\
             --name neo4j \\
             -p 7474:7474 -p 7687:7687 \\
             -e NEO4J_AUTH=neo4j/your_password \\
             neo4j:latest
           ```

        2. **ブラウザで確認**
           - http://localhost:7474 にアクセス
           - 初回ログイン後にパスワードを変更

        3. **接続情報を入力**
           - URL: `bolt://localhost:7687`
           - ユーザー名: `neo4j`
           - パスワード: 設定したパスワード

        ### データベースの選択

        - デフォルトは `neo4j` データベース
        - 複数のプロジェクトを分ける場合は別のDB名を使用可能
        - 新しいDBを作成する場合はNeo4jブラウザで:
          ```cypher
          CREATE DATABASE myproject
          ```

        ### トラブルシューティング

        - **接続できない**: Neo4jが起動しているか確認 `docker ps`
        - **認証エラー**: パスワードが正しいか確認
        - **データベースエラー**: DB名が存在するか確認
        """)

    st.divider()

    return {
        'url': url,
        'username': username,
        'password': password,
        'database': database
    }