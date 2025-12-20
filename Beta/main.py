# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="Crystal Cluster API", version="1.0")

# CrystalCluster インスタンス（グローバル）
cluster_instance = None

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
            "api_configured": False,
            "message": "CrystalCluster not initialized"
        }

    status = cluster_instance.ai_router.get_status()
    return {
        "mode": status["mode"],
        "ollama_available": status["ollama_available"],
        "api_configured": status["api_configured"],
        "ollama_url": status["ollama_url"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)