# bge-m3を自動でロード

import subprocess
import importlib
from sentence_transformers import SentenceTransformer

def ensure_bge_m3():
    try:
        importlib.import_module("sentence_transformers")
    except ImportError:
        subprocess.run(["pip", "install", "sentence-transformers"], check=True)

    model_name = "BAAI/bge-m3"
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        subprocess.run(["pip", "install", "torch"], check=True)
        model = SentenceTransformer(model_name)

    return model
