"""
入力検証ユーティリティ
"""

import re
from typing import Tuple, Optional
import requests
from neo4j import GraphDatabase


def validate_neo4j_connection(
    url: str,
    username: str,
    password: str,
    database: str = "neo4j",
    timeout: int = 5
) -> Tuple[bool, Optional[str]]:
    """
    Neo4j接続をテスト

    Returns:
        (成功: bool, エラーメッセージ: str or None)
    """
    try:
        # URL形式チェック
        if not url.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            return False, "URLは bolt:// または neo4j:// で始まる必要があります"

        # 接続テスト
        driver = GraphDatabase.driver(url, auth=(username, password))

        with driver.session(database=database) as session:
            result = session.run("RETURN 1 AS test")
            result.single()

        driver.close()

        return True, None

    except Exception as e:
        error_msg = str(e)

        # エラーメッセージをわかりやすく
        if "authentication" in error_msg.lower():
            return False, "認証失敗: ユーザー名またはパスワードが正しくありません"
        elif "connection refused" in error_msg.lower():
            return False, "接続失敗: Neo4jサーバーが起動していません"
        elif "database" in error_msg.lower():
            return False, f"データベース '{database}' が見つかりません"
        else:
            return False, f"接続エラー: {error_msg[:100]}"


def validate_openai_api_key(api_key: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    OpenAI APIキーを検証

    Returns:
        (有効: bool, エラーメッセージ: str or None)
    """
    if not api_key:
        return False, "APIキーが空です"

    if not api_key.startswith('sk-'):
        return False, "APIキーは 'sk-' で始まる必要があります"

    try:
        # 簡易的なAPIコール（models.listは軽量）
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout
        )

        if response.status_code == 200:
            return True, None
        elif response.status_code == 401:
            return False, "APIキーが無効です"
        elif response.status_code == 429:
            return False, "レート制限エラー（キーは有効）"
        else:
            return False, f"検証エラー: {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "タイムアウト: APIサーバーに接続できません"
    except Exception as e:
        return False, f"検証エラー: {str(e)[:100]}"


def validate_anthropic_api_key(api_key: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Anthropic APIキーを検証

    Returns:
        (有効: bool, エラーメッセージ: str or None)
    """
    if not api_key:
        return False, "APIキーが空です"

    if not api_key.startswith('sk-ant-'):
        return False, "APIキーは 'sk-ant-' で始まる必要があります"

    try:
        # Anthropic API（簡易チェック）
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "test"}]
            },
            timeout=timeout
        )

        if response.status_code == 200:
            return True, None
        elif response.status_code == 401:
            return False, "APIキーが無効です"
        elif response.status_code == 429:
            return False, "レート制限エラー（キーは有効）"
        else:
            return False, f"検証エラー: {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "タイムアウト: APIサーバーに接続できません"
    except Exception as e:
        return False, f"検証エラー: {str(e)[:100]}"


def validate_ollama_connection(url: str, timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """
    Ollama接続をテスト

    Returns:
        (接続可能: bool, エラーメッセージ: str or None)
    """
    try:
        response = requests.get(f"{url}/api/tags", timeout=timeout)

        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, f"接続成功 ({len(models)} モデル利用可能)"
        else:
            return False, f"接続エラー: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "接続失敗: Ollamaサーバーが起動していません"
    except Exception as e:
        return False, f"エラー: {str(e)[:100]}"


def validate_parameter_range(
    value: float,
    min_val: float,
    max_val: float,
    param_name: str
) -> Tuple[bool, Optional[str]]:
    """
    パラメータの範囲チェック

    Returns:
        (有効: bool, 警告メッセージ: str or None)
    """
    if value < min_val or value > max_val:
        return False, f"{param_name} は {min_val} ～ {max_val} の範囲で指定してください"

    # 推奨範囲外の警告
    recommended_ranges = {
        'entity_linking_threshold': (0.85, 0.92),
        'retrieval_chunk_size': (256, 512),
        'graph_chunk_size': (384, 640),
    }

    if param_name in recommended_ranges:
        rec_min, rec_max = recommended_ranges[param_name]
        if value < rec_min or value > rec_max:
            return True, f"⚠️ 推奨範囲外 ({rec_min} ～ {rec_max})"

    return True, None


def validate_directory(path: str) -> Tuple[bool, Optional[str]]:
    """
    ディレクトリの存在確認

    Returns:
        (存在: bool, エラーメッセージ: str or None)
    """
    from pathlib import Path

    if not path:
        return False, "パスが空です"

    path_obj = Path(path)

    if not path_obj.exists():
        return False, "ディレクトリが存在しません"

    if not path_obj.is_dir():
        return False, "ファイルではなくディレクトリを指定してください"

    return True, None


def validate_file_patterns(patterns: list) -> Tuple[bool, Optional[str]]:
    """
    ファイルパターンの検証

    Returns:
        (有効: bool, エラーメッセージ: str or None)
    """
    if not patterns:
        return False, "最低1つのパターンを指定してください"

    valid_extensions = ['.pdf', '.md', '.txt', '.docx', '.html', '.tei.xml']

    for pattern in patterns:
        # *.pdf 形式のチェック
        if not pattern.startswith('*'):
            return False, f"パターン '{pattern}' は '*' で始まる必要があります"

        # 拡張子チェック
        ext = pattern[1:]  # '*' を除去
        if ext not in valid_extensions:
            return False, f"未対応の拡張子: {ext}"

    return True, None