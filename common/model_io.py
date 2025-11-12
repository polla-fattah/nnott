import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def _to_path(path):
    return Path(path).expanduser().resolve()


def save_model(model, path, metadata: Optional[Dict[str, Any]] = None):
    payload = {
        "state": model.state_dict() if hasattr(model, "state_dict") else None,
        "metadata": metadata or {},
    }
    target = _to_path(path)
    with target.open("wb") as fh:
        pickle.dump(payload, fh)
    return target


def load_model(model, path):
    target = _to_path(path)
    with target.open("rb") as fh:
        payload = pickle.load(fh)
    state = payload.get("state")
    if state is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
    return payload.get("metadata", {})
