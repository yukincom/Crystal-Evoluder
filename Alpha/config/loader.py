#　エキスパートモード
from .presets import PRESETS

def load_config(mode: str, user_values: dict | None = None):
    """
    mode: "preset" or "expert"
    user_values: {'retrieval_chunk_size': 400, ...}
    """
    if mode == "preset":
        preset_name = user_values.get("preset_name")
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset {preset_name}")
        return PRESETS[preset_name]

    if mode == "expert":
        # expert は全部任せる
        required = [
            "retrieval_chunk_size", "retrieval_chunk_overlap",
            "graph_chunk_size", "graph_chunk_overlap",
            "max_triplets_per_chunk",
            "entity_linking_threshold",
            "relation_compat_threshold",
            "final_weight_cutoff",
        ]
        for r in required:
            if r not in user_values:
                raise ValueError(f"Missing required parameter: {r}")

        return user_values

    raise ValueError("mode must be 'preset' or 'expert'")
