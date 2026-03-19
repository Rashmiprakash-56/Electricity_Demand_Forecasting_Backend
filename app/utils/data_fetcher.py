"""
Fetches dataset CSV files from Hugging Face Hub on application startup.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

_DATASET_FILES = ["energy_dataset.csv", "weather_features.csv"]


def fetch_dataset_from_hub() -> None:
    """
    Download dataset files from Hugging Face Hub into DATA_DIR.

    Skips files that already exist locally so the download only
    happens once per container / environment lifecycle.
    """
    repo_id = settings.HF_DATASET_REPO
    if not repo_id:
        log.warning(
            "HF_DATASET_REPO is not configured. "
            "Assuming data files already exist locally."
        )
        return

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in _DATASET_FILES:
        dest = settings.DATA_DIR / filename

        if dest.exists():
            log.info(f"Dataset file already present, skipping download: {dest}")
            continue

        log.info(f"Downloading {filename} from HF dataset '{repo_id}' ...")
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=settings.HF_TOKEN or None,
                local_dir=str(settings.DATA_DIR),
            )
            log.info(f"Downloaded {filename} → {cached_path}")
        except Exception as e:
            log.error(f"Failed to download {filename} from HF Hub: {e}")
            if not dest.exists():
                raise RuntimeError(
                    f"Dataset file '{filename}' is missing and could not "
                    f"be downloaded from '{repo_id}'. Cannot start."
                ) from e
