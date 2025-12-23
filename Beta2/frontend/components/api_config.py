"""
APIキー管理コンポーネント
"""

import streamlit as st
import requests
from utils.config_manager import get_config_manager
from utils.validators import (
    validate_openai_api_key,
    validate_anthropic_api_key,
    validate_ollama_connection
)


def render_api_config():
    """APIキー設定UIを描画（改善版）"""

    st.subheader("🤖 AI設定")

    config_mgr = get_config_manager()

    # ========================================
    # モデル選択（統合UI）
    # ========================================

    st.markdown("### モデル選択")

    current_mode = config_mgr.get('ai', 'mode', 'api')

    # ローカルAIの検出
    available_local_models = _detect_local_models(config_mgr.get('ai', 'ollama_url'))
    has_local = len(available_local_models) > 0

    col1, col2 = st.columns([1, 3])

    with col1:
        # ラジオボタンで選択
        mode = st.radio(
            "AI動作モード",
            options=['local', 'api'],
            format_func=lambda x: {
                'local': '🖥️ Local AI',
                'api': '🌐 API'
            }[x],
            index=0 if current_mode == 'ollama' and has_local else 1,
            key="ai_mode_radio"
        )

    with col2:
        if mode == 'local':
            _render_local_model_selector(config_mgr, available_local_models)
        else:
            _render_api_model_selector(config_mgr)

    # モードをconfigに反映（local→ollama変換）
    config_mgr.set('ai', 'mode', 'ollama' if mode == 'local' else 'api')

    st.divider()

    # ========================================
    # APIキー（API選択時のみ）
    # ========================================

    if mode == 'api':
        _render_api_key_input(config_mgr)
        st.divider()

    # ========================================
    # 図表解析モデル
    # ========================================

    _render_vision_model_selector(config_mgr, mode, available_local_models)

    # 保存ボタン
    st.divider()

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("💾 設定を保存", use_container_width=True, type="primary"):
            if config_mgr.save_config():
                st.success("✅ AI設定を保存しました")
            else:
                st.error("❌ 保存に失敗しました")

    with col2:
        if st.button("🔄", help="デフォルトに戻す"):
            # AI設定のみリセット
            config_mgr.config['ai'] = config_mgr.DEFAULT_CONFIG['ai'].copy()
            st.rerun()


def _detect_local_models(ollama_url: str) -> dict:
    """
    インストール済みのOllamaモデルを検出

    Returns:
        {
            'llm': [{'name': 'llama3.1:70b', 'size': 40, 'capable': True}, ...],
            'vision': [{'name': 'granite3.2-vision', 'size': 2.4, 'capable': True}, ...]
        }
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)

        if response.status_code != 200:
            return {'llm': [], 'vision': []}

        models_data = response.json().get('models', [])

        llm_models = []
        vision_models = []

        for model in models_data:
            name = model.get('name', '')
            size_bytes = model.get('size', 0)
            size_gb = size_bytes / (1024 ** 3)

            # サイズから能力を推定（70B = 約40GB）
            is_capable = size_gb >= 20  # 70B未満を非推奨

            # Vision系とLLM系を分類
            if any(keyword in name.lower() for keyword in ['vision', 'llava', 'granite']):
                vision_models.append({
                    'name': name,
                    'size': round(size_gb, 1),
                    'capable': True  # Visionは能力制限なし
                })
            else:
                llm_models.append({
                    'name': name,
                    'size': round(size_gb, 1),
                    'capable': is_capable
                })

        return {'llm': llm_models, 'vision': vision_models}

    except Exception:
        return {'llm': [], 'vision': []}


def _render_local_model_selector(config_mgr, available_models):
    """ローカルモデル選択UI"""

    llm_models = available_models['llm']

    if not llm_models:
        st.warning("""
        ⚠️ **ローカルモデルが見つかりません**

        Ollamaをインストールしてモデルをダウンロードしてください：
        ```bash
        ollama pull llama3.1:70b
        ```
        """)
        # APIモードへの切り替えを提案
        st.info("💡 APIモードに切り替えることをお勧めします")
        return

    # 能力別に分類
    capable_models = [m for m in llm_models if m['capable']]
    weak_models = [m for m in llm_models if not m['capable']]

    # モデル選択
    current_model = config_mgr.get('ai', 'llm_model', '')

    # selectboxのオプション作成
    model_options = []
    model_display = {}

    for model in capable_models:
        model_options.append(model['name'])
        model_display[model['name']] = f"{model['name']} ({model['size']}GB) ✅"

    for model in weak_models:
        model_options.append(model['name'])
        model_display[model['name']] = f"{model['name']} ({model['size']}GB) ⚠️ 非推奨"

    if not model_options:
        st.error("利用可能なモデルがありません")
        return

    # デフォルト選択（現在の設定 or 最初の有能なモデル）
    default_index = 0
    if current_model in model_options:
        default_index = model_options.index(current_model)

    selected_model = st.selectbox(
        "LLMモデル",
        options=model_options,
        index=default_index,
        format_func=lambda x: model_display[x],
        help="70B以上のモデルを推奨（40GB以上）",
        key="local_llm_model"
    )

    config_mgr.set('ai', 'llm_model', selected_model)

    # 警告表示
    selected_info = next((m for m in llm_models if m['name'] == selected_model), None)
    if selected_info and not selected_info['capable']:
        st.warning("""
        ⚠️ **非推奨モデル**

        このモデルは性能が不十分な可能性があります。
        高品質な結果を得るには70B以上のモデル（40GB+）を使用してください。
        """)

    # APIキー入力欄（グレーアウト）
    st.text_input(
        "APIキー",
        value="（ローカルモードでは不要）",
        disabled=True,
        help="ローカルモードではAPIキーは使用しません",
        key="local_api_key_disabled"
    )


def _render_api_model_selector(config_mgr):
    """APIモデル選択UI"""

    current_model = config_mgr.get('ai', 'llm_model', 'gpt-4o-mini')

    llm_model = st.text_input(
        "LLMモデル",
        value=current_model,
        placeholder="gpt-4o-mini",
        help="💡 GPT-4o-mini以上を推奨",
        key="api_llm_model"
    )

    config_mgr.set('ai', 'llm_model', llm_model)

    # 推奨モデルのヒント
    st.caption("""
    📝 **推奨モデル**
    OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`
    Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
    """)

    # コスト警告
    if 'gpt-4' in llm_model and 'mini' not in llm_model:
        st.warning("⚠️ GPT-4（非mini）は高コストです。大量処理には注意してください。")


def _render_api_key_input(config_mgr):
    """APIキー入力UI"""

    st.markdown("### APIキー")

    col1, col2 = st.columns([4, 1])

    with col1:
        # 現在のLLMモデルからプロバイダーを推定
        current_model = config_mgr.get('ai', 'llm_model', '')

        if 'claude' in current_model.lower():
            provider = 'anthropic'
            placeholder = "sk-ant-..."
            help_text = "AnthropicのAPIキー"
            link = "https://console.anthropic.com/"
        else:
            provider = 'openai'
            placeholder = "sk-..."
            help_text = "OpenAIのAPIキー"
            link = "https://platform.openai.com/api-keys"

        api_key = st.text_input(
            "Your API Key",
            value=config_mgr.get_api_key(provider),
            type="password",
            placeholder=placeholder,
            help=help_text,
            key=f"{provider}_api_key_input"
        )

        config_mgr.set_api_key(provider, api_key)

    with col2:
        st.write("")
        st.write("")
        if st.button("検証", key=f"verify_{provider}"):
            if not api_key:
                st.warning("⚠️ APIキーを入力してください")
            else:
                with st.spinner("検証中..."):
                    if provider == 'openai':
                        success, error_msg = validate_openai_api_key(api_key)
                    else:
                        success, error_msg = validate_anthropic_api_key(api_key)

                    if success:
                        st.success("✅ 有効")
                    else:
                        st.error(f"❌ {error_msg}")

    st.caption(f"🔗 [APIキーを取得]({link})")


def _render_vision_model_selector(config_mgr, mode, available_models):
    """図表解析モデル選択UI"""

    st.markdown("### 図表解析")

    if mode == 'local':
        vision_models = available_models['vision']

        if not vision_models:
            st.warning("""
            ⚠️ **Visionモデルが見つかりません**

            図表解析を使用するにはVisionモデルをインストールしてください：
            ```bash
            ollama pull granite3.2-vision
            ```
            """)
            # 図表解析を無効化
            config_mgr.set('figure_analysis', 'enable', False)
            st.info("図表解析が無効化されました")
            return

        # モデル選択
        model_options = [m['name'] for m in vision_models]
        current_vision = config_mgr.get('ai', 'vision_model', '')

        default_index = 0
        if current_vision in model_options:
            default_index = model_options.index(current_vision)

        vision_model = st.selectbox(
            "Visionモデル",
            options=model_options,
            index=default_index,
            format_func=lambda x: f"{x if ':' in x else x + ':latest'} ({next((m['size'] for m in vision_models if m['name']==x), '?')}GB)",
            key="local_vision_model"
        )

        config_mgr.set('ai', 'vision_model', vision_model)
        config_mgr.set('figure_analysis', 'enable', True)

    else:
        # API版ではVisionモデルは不要（図表解析をOllamaで実行）
        st.info("""
        💡 **図表解析について**

        APIモード時は、図表解析にローカルのOllama（granite3.2-vision）を使用します。
        Ollamaをインストールしていない場合は図表解析が無効化されます。
        """)

        # Ollamaの有効性確認
        ollama_url = config_mgr.get('ai', 'ollama_url', 'http://localhost:11434')
        success, msg = validate_ollama_connection(ollama_url)

        if success:
            st.success(f"✅ Ollama接続OK：{msg}")
            config_mgr.set('figure_analysis', 'enable', True)
        else:
            st.warning(f"⚠️ Ollama未接続：図表解析が無効化されます")
            config_mgr.set('figure_analysis', 'enable', False)