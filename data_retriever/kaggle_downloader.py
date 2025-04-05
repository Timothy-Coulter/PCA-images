import os
import shutil
import logging
from .base_downloader import BaseDownloader

log = logging.getLogger(__name__)

class KaggleDownloader(BaseDownloader):
    """Downloads datasets from Kaggle using the Kaggle API."""
    def __init__(self, dataset_name, output_dir, download_dir, kaggle_id, task_type):
        super().__init__(dataset_name, output_dir, download_dir)
        self.kaggle_id = kaggle_id
        self.task_type = task_type
        self.kaggle_api = self._setup_kaggle_api()

    def _setup_kaggle_api(self):
        """Authenticates with the Kaggle API."""
        try:
            # Import dynamically to avoid hard dependency if not used
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            log.info("Kaggle API authenticated successfully.")
            return api
        except ImportError:
            log.error("Kaggle API client not installed. Please run 'pip install kaggle'.")
            return None
        except Exception as e:
            # Catching a broad exception might hide specific auth issues,
            # but Kaggle API errors can be varied.
            log.error(f"Kaggle API authentication failed. Ensure kaggle.json is configured correctly in ~/.kaggle/ or via environment variables: {e}")
            return None

    def download(self):
        """Downloads and extracts files using the Kaggle API."""
        if not self.kaggle_api:
            return None

        log.info(f"Downloading dataset '{self.kaggle_id}' from Kaggle to {self.download_dir}...")
        try:
            # kaggle datasets download [-f <file_name>] -d <dataset_id> -p <path> --unzip
            # The API call handles both download and extraction if unzip=True
            self.kaggle_api.dataset_download_files(self.kaggle_id, path=self.download_dir, unzip=True, quiet=False) # quiet=False shows progress
            log.info(f"Kaggle dataset '{self.kaggle_id}' downloaded and extracted.")
            # The files are extracted directly into download_dir.
            return self.download_dir # Source for organization is the download dir itself
        except Exception as e:
            # The Kaggle API can raise various exceptions (e.g., for 404, 403, rate limits)
            log.error(f"Error downloading dataset '{self.kaggle_id}' from Kaggle: {e}")
            return None

    def organize(self, source_path):
        """
        Generic organization for Kaggle datasets: copies contents to 'all_data'.
        Args:
            source_path (str): The directory where Kaggle downloaded and extracted files (self.download_dir).
        """
        # Kaggle datasets vary wildly in structure. Apply generic copy.
        log.info(f"Performing generic organization for Kaggle dataset from {source_path}")
        if not source_path or not os.path.isdir(source_path):
             log.error(f"Source path '{source_path}' is not a valid directory for organization.")
             return False

        os.makedirs(self.dataset_final_path, exist_ok=True)
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        # Ensure target doesn't exist if source is copied directly
        if os.path.exists(target_all_data):
             log.warning(f"Target 'all_data' directory exists. Removing before copy: {target_all_data}")
             shutil.rmtree(target_all_data)

        try:
            # Copy contents of source_path (download_dir) into target_all_data
            # Avoid copying the final dataset dir into itself if output is subdir of download
            # Use copytree for simplicity
            shutil.copytree(source_path, target_all_data, ignore=shutil.ignore_patterns(os.path.basename(self.dataset_final_path)))
            log.info(f"Copied contents of {source_path} to {target_all_data}")
            log.warning("Generic organization used for Kaggle dataset. Manual adjustment or a specific organizer might be needed based on dataset structure.")
            return True
        except Exception as e:
            log.error(f"Error during generic organization for Kaggle dataset: {e}", exc_info=True)
            return False