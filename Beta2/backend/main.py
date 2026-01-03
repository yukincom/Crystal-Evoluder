# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import requests
    
from pathlib import Path


app = FastAPI(title="Crystal Cluster API", version="1.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Viteのデフォルトポート
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# 設定管理用のPydanticモデル
# ========================================

class Neo4jConfig(BaseModel):
    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

class AIConfig(BaseModel):
    mode: str = "api"  # "api" or "ollama"
    ollama_url: str = "http://localhost:11434"
    
    # モード別のモデル名（独立管理）
    api_model: str = "gpt-4o-mini"
    ollama_model: str = ""
    
    # 後方互換性のため残す（非推奨）
    # llm_model: Optional[str] = None
    
    api_key: str = ""
    # vision_model: str = "granite3.2-vision"
    # 品質チェック専用のオーバーライド設定
    quality_mode: Optional[str] = None
    quality_check_ollama_model: Optional[str] = None
    quality_check_api_model: Optional[str] = None
    quality_check_api_key: Optional[str] = None

    # Refiner専用のオーバーライド設定
    refiner_mode: Optional[str] = None
    refiner_api_key: Optional[str] = None
    refiner_ollama_model: Optional[str] = None
    refiner_api_model: Optional[str] = None

class ParametersConfig(BaseModel):
    entity_linking_threshold: float = 0.88
    retrieval_chunk_size: int = 320
    retrieval_chunk_overlap: int = 120
    graph_chunk_size: int = 512
    graph_chunk_overlap: int = 50
    relation_compat_threshold: float = 0.11
    final_weight_cutoff: float = 0.035
    max_triplets_per_chunk: int = 15


class SelfRAGConfig(BaseModel):
    enable: bool = True
    confidence_threshold: float = 0.75
    # Refinerだけオーバーライド可能（Noneならメインを使う）
    refiner_model: Optional[str] = None
    max_retries: int = 1
    token_budget: int = 100000

class ProcessingConfig(BaseModel):
    input_dir: str = ""
    output_dir: str = "./output"
    patterns: List[str] = ["*.pdf", "*.md", "*.docx"]

class AdvancedConfig(BaseModel):
    """フロントエンド用の統合設定"""
    # チャンク設定（ParametersConfigから）
    graph_chunk_size: int = 512
    graph_chunk_overlap: int = 50
    retrieval_chunk_size: int = 320
    retrieval_chunk_overlap: int = 120
    
    # エンティティリンキング
    entity_linking_threshold: float = 0.88
    relation_compat_threshold: float = 0.11
    final_weight_cutoff: float = 0.035
    max_triplets_per_chunk: int = 15

    # Self-RAG設定
    self_rag_enabled: bool = True
    self_rag_threshold: float = 0.75
    
    self_rag_max_retries: int = 1
    self_rag_token_budget: int = 100000

class FullConfig(BaseModel):
    neo4j: Neo4jConfig
    ai: AIConfig
    parameters: ParametersConfig
    self_rag: SelfRAGConfig
    processing: ProcessingConfig
    advanced: AdvancedConfig

# ========================================
# 設定の保存/読込
# ========================================

CONFIG_FILE = Path(__file__).parent / "config" / "user_config.json"

def load_config() -> FullConfig:
    """設定ファイルから読み込み"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            config = FullConfig(**data)
    else:
        config = FullConfig(
            neo4j=Neo4jConfig(),
            ai=AIConfig(),
            parameters=ParametersConfig(),
            self_rag=SelfRAGConfig(),
            processing=ProcessingConfig(),
            advanced=AdvancedConfig()
        )

    # Ollamaモデルの自動検出と検証
    if config.ai.mode == 'ollama':
        try:
            response = requests.get(
                f"{config.ai.ollama_url}/api/tags",
                timeout=3
            )
            
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                ollama_models = []
                
                for model in models_data:
                    name = model.get('name', '')
                    # Visionモデルを除外
                    if not any(kw in name.lower() for kw in ['vision', 'llava', 'granite']):
                        ollama_models.append(name)
                
                # ollama_modelが未設定または無効な場合、自動設定
                if not config.ai.ollama_model or config.ai.ollama_model not in ollama_models:
                    if ollama_models:
                        config.ai.ollama_model = ollama_models[0]  # 最初のモデルを使用
                        print(f"✅ Ollama model auto-detected: {config.ai.ollama_model}")
                    else:
                        print("⚠️ No Ollama LLM models found")
                        config.ai.ollama_model = ""
                
        except Exception as e:
            print(f"⚠️ Could not connect to Ollama: {e}")
            config.ai.ollama_model = ""
    
    # advanced セクション同期（フロントエンド表示用）
    adv = config.advanced
    
    # Self-RAGの基本設定も同期
    adv.self_rag_enabled = config.self_rag.enable
    adv.self_rag_threshold = config.self_rag.confidence_threshold
    adv.self_rag_max_retries = config.self_rag.max_retries
    adv.self_rag_token_budget = config.self_rag.token_budget
    
    # パラメータも同期
    adv.graph_chunk_size = config.parameters.graph_chunk_size
    adv.graph_chunk_overlap = config.parameters.graph_chunk_overlap
    adv.retrieval_chunk_size = config.parameters.retrieval_chunk_size
    adv.retrieval_chunk_overlap = config.parameters.retrieval_chunk_overlap
    adv.entity_linking_threshold = config.parameters.entity_linking_threshold
    adv.relation_compat_threshold = config.parameters.relation_compat_threshold
    adv.final_weight_cutoff = config.parameters.final_weight_cutoff
    adv.max_triplets_per_chunk = config.parameters.max_triplets_per_chunk
    
    return config

def save_config(config: FullConfig):
    """設定ファイルに保存"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)

# ========================================
# 設定管理エンドポイント
# ========================================
# main.py の設定管理エンドポイント部分

# ========================================
# 設定管理エンドポイント
# ========================================

@app.get("/config")
async def get_config():
    """現在の設定を取得"""
    config = load_config()
    return config.model_dump()

@app.post("/config")
async def update_config(config: FullConfig):
    """
    設定を更新（検証付き）
    
    フロントエンドから送られた設定を保存し、
    実行中のAIRouterインスタンスがあれば即座に更新する
    """
    try:
        # 1. ファイルに保存
        save_config(config)
        
        # 2. グローバルインスタンスがあればAIRouterを更新
        global cluster_instance
        if cluster_instance and hasattr(cluster_instance, 'ai_router'):
            # 新しい設定でAIRouterを更新
            new_ai_config = config.ai.model_dump()
            cluster_instance.ai_router.update_config(new_ai_config)
            
            return {
                "status": "ok",
                "message": "✅ 設定を保存し、AIルーターを更新しました"
            }
        
        # インスタンスがない場合は保存のみ
        return {
            "status": "ok",
            "message": "✅ 設定を保存しました（次回起動時に反映されます）"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"❌ 保存失敗: {str(e)}"
        )

@app.post("/config/neo4j/test")
async def test_neo4j_connection(neo4j: Neo4jConfig):
    """Neo4j接続をテスト"""
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            neo4j.url,
            auth=(neo4j.username, neo4j.password)
        )
        
        with driver.session(database=neo4j.database) as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        
        driver.close()
        
        return {"status": "ok", "message": "✅ Connection successful"}
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Connection failed: {str(e)}"
        )

@app.get("/config/ollama/models")
async def get_ollama_models():
    """Ollamaのインストール済みモデルを取得"""
    config = load_config()
    ollama_url = config.ai.ollama_url
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code != 200:
            return {"models": [], "available": False}

        models_data = response.json().get('models', [])
        models = []

        for model in models_data:
            name = model.get('name', '')
            size_bytes = model.get('size', 0)
            size_gb = round(size_bytes / (1024 ** 3), 1)

            # ✅ 3段階判定
            usable = size_gb >= 4                    # 最低限使える（7B以上）
            recommended_for_quality = size_gb >= 4   # Quality用推奨
            recommended_for_base = size_gb >= 8      # Base用推奨（14B以上）

            is_vision = any(kw in name.lower() for kw in ['vision', 'llava', 'granite'])

            models.append({
                'name': name,
                'size': size_gb,
                'capable': usable,
                'is_vision': is_vision,
                'recommended_for_base': recommended_for_base,
                'recommended_for_quality': recommended_for_quality
            })
        
        return {
            "models": models,
            "available": True,
            "url": ollama_url
        }
    
    except Exception as e:
        return {
            "models": [],
            "available": False,
            "error": str(e)
        }

@app.post("/test_ai")
async def test_ai_connection(ai_config: dict):
    """
    AI接続をテスト（API または Ollama）
    
    Args:
        ai_config: {
            mode: 'api' | 'ollama',
            api_key: str (API mode時),
            api_model: str (API mode時),
            ollama_model: str (Ollama mode時),
            ollama_url: str (Ollama mode時)
        }
    
    Returns:
        {success: bool, message: str}
    """
    mode = ai_config.get('mode', 'api')
    
    if mode == 'api':
        # APIキーの検証
        api_key = ai_config.get('api_key', '')
        api_model = ai_config.get('api_model', '')
        
        if not api_key:
            raise HTTPException(status_code=400, detail="APIキーが設定されていません")
        
        # プロバイダーを判定
        if 'claude' in api_model.lower():
            provider = 'anthropic'
        else:
            provider = 'openai'
        
        # 既存のvalidate_api_keyを利用
        try:
            result = await api_key(provider, api_key)
            return {
                "success": True,
                "message": f"✅ {provider.upper()} 接続成功！モデル: {api_model}"
            }
        except HTTPException as e:
            return {
                "success": False,
                "message": f"❌ {provider.upper()} 接続失敗: {e.detail}"
            }
    
    elif mode == 'ollama':
        # Ollama接続確認
        ollama_url = ai_config.get('ollama_url', 'http://localhost:11434')
        ollama_model = ai_config.get('ollama_model', '')
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "message": f"❌ Ollama接続失敗（ステータス: {response.status_code}）"
                }
            
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # 指定されたモデルが存在するか確認
            if ollama_model and ollama_model not in model_names:
                return {
                    "success": False,
                    "message": f"❌ モデル '{ollama_model}' が見つかりません"
                }
            
            # 指定されたモデルの能力チェック
            if ollama_model:
                target_model = next((m for m in models if m.get('name') == ollama_model), None)
                if target_model:
                    size_bytes = target_model.get('size', 0)
                    size_gb = round(size_bytes / (1024 ** 3), 1)
                    
                    return {
                        "success": True,
                        "message": f"✅ Ollama接続成功！モデル: {ollama_model} ({size_gb}GB)"
                    }
            
            # モデル未指定の場合
            return {
                "success": True,
                "message": "✅ Ollama接続成功！"
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "message": "❌ Ollamaに接続できません。Ollamaが起動しているか確認してください。"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ エラー: {str(e)}"
            }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

# ========================================
# 既存のエンドポイント（残す）
# ========================================

cluster_instance = None

@app.get("/")
def root():
    return {"message": "Crystal Cluster API", "version": "1.0"}

@app.get("/ai_status")
def get_ai_status():
    """AI設定とOllama接続状況"""
    config = load_config()
    
    # Ollama接続確認
    ollama_available = False
    try:
        response = requests.get(
            f"{config.ai.ollama_url}/api/tags",
            timeout=2
        )
        ollama_available = response.status_code == 200
    except:
        pass
    
    return {
        "mode": config.ai.mode,
        "ollama_available": ollama_available,
        "ollama_url": config.ai.ollama_url,
        "api_configured": bool(config.ai.api_key),
        "models": {
            "llm": config.ai.ollama_model,
            "vision": config.ai.vision_model
        }
    }

# ... 既存のエンドポイント（setup, switch_mode等）もそのまま残す ...

# CrystalCluster インスタンス（グローバル）
# cluster_instance = None  # 既に定義済み

# 設定ファイルパス
# CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")  # 古い設定は削除

class ConfigRequest(BaseModel):
    mode: str  # "preset" or "expert"
    values: Dict[str, Any]

class SetupResponse(BaseModel):
    status: str
    config: Dict[str, Any]
    warnings: Optional[list] = None

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

        if config.get('ai_routing', {}).get('mode') == 'api':
            model = config.get('api_model', 'gpt-4o-mini')
        else:
            model = config.get('ollama_model', '')

        if not config.get('self_rag_critic_model'):
            config['self_rag_critic_model'] = model
        
        # Refinerが未設定なら基本モデルに追従
        if not config.get('self_rag_refiner_model'):
            config['self_rag_refiner_model'] = model

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)