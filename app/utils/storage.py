"""
Hugging Face Hub integration for model artifact persistence.

In development (no HF_TOKEN set):  save/load uses local filesystem only.
In production  (HF_TOKEN set):     syncs artifacts to/from HF Hub directly.
"""

from pathlib import Path
from typing import Any
import joblib
import pickle
import tempfile
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)


def _is_configured() -> bool:
    """Check if HF Hub credentials are available."""
    return bool(settings.HF_TOKEN and settings.HF_MODEL_REPO)


def save_to_hub_or_local(obj: Any, filename: str, local_fallback_path: Path, use_pickle: bool = False):
    """
    Saves an object (model, encoder, explainer) directly to HF Hub if configured.
    Falls back to local file path if HF Hub is not configured or fails.
    """
    if _is_configured():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=settings.HF_TOKEN)
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_path = tmp.name
                
            try:
                if use_pickle:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(obj, f)
                else:
                    joblib.dump(obj, temp_path)
                    
                api.upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=filename,
                    repo_id=settings.HF_MODEL_REPO,
                    repo_type="model",
                )
                log.info(f"Uploaded artifact → HF Hub: {filename}")
                return
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            log.error(f"Upload to HF Hub failed: {e}. Falling back to local storage.")
    
    # Local fallback
    log.info(f"HF Hub not used/configured. Saving locally to {local_fallback_path}")
    Path(local_fallback_path).parent.mkdir(parents=True, exist_ok=True)
    if use_pickle:
        with open(local_fallback_path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        joblib.dump(obj, local_fallback_path)


def load_from_hub_or_local(filename: str, local_fallback_path: Path, use_pickle: bool = False) -> Any:
    """
    Loads an object directly from HF Hub cache if configured.
    Falls back to local file path if HF Hub is not configured or fails.
    """
    if _is_configured():
        try:
            from huggingface_hub import hf_hub_download
            cached_path = hf_hub_download(
                repo_id=settings.HF_MODEL_REPO,
                filename=filename,
                repo_type="model",
                token=settings.HF_TOKEN,
            )
            log.info(f"Loaded artifact ← HF Hub: {filename}")
            
            if use_pickle:
                with open(cached_path, 'rb') as f:
                    return pickle.load(f)
            else:
                return joblib.load(cached_path)
                
        except Exception as e:
            log.warning(f"Download from HF Hub failed: {e}. Trying local fallback.")
    
    # Local fallback
    if Path(local_fallback_path).exists():
        log.info(f"Loading artifact from local check: {local_fallback_path}")
        if use_pickle:
            with open(local_fallback_path, 'rb') as f:
                return pickle.load(f)
        else:
            return joblib.load(local_fallback_path)
    
    raise FileNotFoundError(f"Artifact {filename} not found on HF Hub or locally at {local_fallback_path}")
