"""Script to download the WISDM dataset."""

import logging
import sys
import tarfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WISDM_URL = (
    "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
)
DATA_DIR = Path(__file__).parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"


def download_wisdm():
    """Download and extract WISDM dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_file = DATA_DIR / "WISDM_ar_latest.tar.gz"

    # Download the archive
    logger.info(f"Downloading WISDM dataset archive to {archive_file}")
    try:
        response = requests.get(WISDM_URL, stream=True)
        response.raise_for_status()

        with open(archive_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Download completed successfully")

        # Extract the archive
        logger.info(f"Extracting archive to {RAW_DIR}")
        with tarfile.open(archive_file, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)

        # Clean up
        archive_file.unlink()
        logger.info("Extraction completed and archive cleaned up")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)
    except tarfile.TarError as e:
        logger.error(f"Error extracting archive: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_wisdm()
