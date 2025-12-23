# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os

app = FastAPI(title="Crystal Cluster API", version="1.0")

# CORS設定を追加（Reactからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React開発サーバー
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CrystalCluster インスタンス（グローバル）
cluster_instance = None

# 設定ファイルパス
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

class ConfigRequest(BaseModel):
    mode: str  # "preset" or "expert"
    values: Dict[str, Any]

class SetupResponse(BaseModel):
    status: str
    config: Dict[str, Any]
    warnings: Optional[list] = None

@app.get("/")
def root():
    return {"message": "Crystal Cluster API", "version": "1.0"}

@app.get("/parameters")
def get_parameters(param_name: Optional[str] = None):
    """パラメータ情報を取得"""
    from config import get_parameter_info
    return get_parameter_info(param_name)

@app.get("/presets")
def get_presets():
    """推奨プリセット一覧を取得"""
    from config import RECOMMENDED_PRESETS
    return {
        "presets": list(RECOMMENDED_PRESETS.keys()),
        "details": RECOMMENDED_PRESETS
    }

@app.post("/setup", response_model=SetupResponse)
def setup(req: ConfigRequest):
    """
    設定をロードしてモデルを初期化
    """
    from config import load_config
    from core.crystal_cluster import CrystalCluster

    global cluster_instance

    try:
        # 設定ロード
        config = load_config(req.mode, req.values)

        # CrystalCluster初期化
        cluster_instance = CrystalCluster(custom_config=config)

        return SetupResponse(
            status="ok",
            config=config,
            warnings=config.get('warnings')
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build_graph")
def build_graph(documents: list, config: dict):
    """
    グラフを構築（実装例）
    """
    # TODO: CrystalClusterを呼び出し
    return {"status": "not_implemented"}

@app.post("/switch_mode")
def switch_ai_mode(mode: str):
    """
    AI動作モードを切り替え

    Args:
        mode: "api" または "ollama"

    Returns:
        新しい設定
    """
    global cluster_instance

    if cluster_instance is None:
        raise HTTPException(status_code=400, detail="CrystalCluster not initialized")

    try:
        success = cluster_instance.ai_router.switch_mode(mode)
        if success:
            return {
                "status": "success",
                "mode": cluster_instance.ai_router.mode,
                "message": f"Switched to {mode} mode"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch to {mode} mode")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai_status")
def get_ai_status():
    """
    現在のAI設定とOllama接続状況を返す

    Returns:
        {
            'mode': 'api' or 'ollama',
            'ollama_available': bool,
            'api_configured': bool
        }
    """
    global cluster_instance

    if cluster_instance is None:
        # デフォルト設定を返す
        return {
            "mode": "api",
            "ollama_available": False,
            "ollama_url": "http://localhost:11434",
            "models": {
                "triplet_extraction": "gpt-4o-mini",
                "quality_check": "gpt-4o-mini",
                "self_rag": "gpt-4o-mini"
            },
            "message": "CrystalCluster not initialized"
        }

    status = cluster_instance.ai_router.get_status()
    return {
        "mode": status["mode"],
        "ollama_available": status["ollama_available"],
        "ollama_url": status["ollama_url"],
        "models": status["models"]
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile]):
    """
    ファイルアップロード

    Returns:
        {'status': 'ok', 'file_count': int}
    """
    # TODO: ファイル処理の実装
    file_count = len(files)
    # ここでファイルを保存したり処理したりする
    return {"status": "ok", "file_count": file_count}

@app.get("/config")
def get_config():
    """
    設定を取得

    Returns:
        設定辞書
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # デフォルト設定を返す
            return {
                "neo4j": {
                    "url": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "",
                    "database": "neo4j"
                },
                "ai": {
                    "mode": "api",
                    "api_key": "",
                    "model": "gpt-4o-mini",
                    "local_models": [],
                    "figure_model": "granite"
                },
                "processing": {
                    "input_dir": "",
                    "output_dir": ""
                },
                "advanced": {
                    "graph_chunk_size": 512,
                    "retrieval_chunk_size": 320,
                    "self_rag_enabled": True,
                    "self_rag_threshold": 0.75
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
def save_config(config: Dict[str, Any]):
    """
    設定を保存

    Args:
        config: 設定辞書

    Returns:
        {'status': 'ok'}
    """
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_neo4j")
def test_neo4j_connection(config: Dict[str, Any]):
    """
    Neo4j接続テスト

    Args:
        config: Neo4j設定

    Returns:
        {'status': 'ok' | 'error', 'message': str}
    """
    try:
        # TODO: 実際のNeo4j接続テストを実装
        # ここではダミー
        return {"status": "ok", "message": "Connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/test_ai")
def test_ai_connection(config: Dict[str, Any]):
    """
    AI接続テスト

    Args:
        config: AI設定

    Returns:
        {'status': 'ok' | 'error', 'message': str}
    """
    try:
        # TODO: 実際のAI接続テストを実装
        # ここではダミー
        return {"status": "ok", "message": "Connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)