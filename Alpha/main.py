# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from config.loader import load_config
from model.embed import ensure_bge_m3
from model.triplet import TripletExtractor

app = FastAPI()

class ConfigRequest(BaseModel):
    mode: str  # preset / expert
    values: dict

@app.post("/setup")
def setup(req: ConfigRequest):
    config = load_config(req.mode, req.values)

    embed_model = ensure_bge_m3()

    extractor = TripletExtractor(
        mode=req.values.get("triplet_mode", "api"),
        model_name=req.values.get("triplet_model", "gpt-4o-mini"),
        api_key=req.values.get("api_key")
    )

    return {
        "status": "ok",
        "config": config,
        "embedding_model": "bge-m3",
        "triplet_mode": extractor.mode
    }
