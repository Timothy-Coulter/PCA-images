import os
import shutil
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders."""
    def __init__(self, dataset_name, output_dir, download_dir):
        self.dataset_name = dataset_name
        self.output_dir = os.path.abspath(output_dir)
        self.download_dir = os.path.abspath(download_dir)
        # Adjust final path if it conflicts with download dir name
        final_path_base = os.path.join(self.output_dir, self.dataset_name)
        if final_path_base == self.download_dir:
             self.dataset_final_path = final_path_base + "_organized"
             log.warning(f"Dataset name '{self.dataset_name}' conflicts with download directory name. Final data will be in: {self.dataset_final_path}")
        else:
             self.dataset_final_path = final_path_base


    @abstractmethod
    def download(self):
        """
        Downloads the raw dataset files.
        Returns:
            str or None: Path to the downloaded content (could be a folder or file)
                         if successful, None otherwise.
        """
        pass

    @abstractmethod
    def organize(self, source_path):
        """
        Organizes the downloaded data from source_path into the final structure
        at self.dataset_final_path.
        Args:
            source_path (str): The path to the downloaded content returned by download().
        Returns:
            bool: True on success, False otherwise.
        """
        pass

    def retrieve(self, force=False):
        """
        Downloads and organizes the dataset.
        Args:
            force (bool): If True, redownload and organize even if the final directory exists.
        Returns:
            bool: True if the dataset is successfully retrieved or already exists, False otherwise.
        """
        if not force and os.path.exists(self.dataset_final_path):
            log.warning(f"Dataset directory '{self.dataset_final_path}' already exists. Skipping retrieval. Use --force to overwrite.")
            return True # Consider it success if already exists

        if force and os.path.exists(self.dataset_final_path):
             log.warning(f"Force flag set. Removing existing dataset directory: {self.dataset_final_path}")
             try:
                  shutil.rmtree(self.dataset_final_path)
             except OSError as e:
                  log.error(f"Error removing existing directory {self.dataset_final_path}: {e}")
                  return False # Cannot proceed if removal fails

        log.info(f"Starting dataset retrieval for: {self.dataset_name}")
        log.info(f"Target directory: {self.dataset_final_path}")
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output base exists

        downloaded_source_path = None
        organization_success = False
        try:
            downloaded_source_path = self.download()

            if not downloaded_source_path or not os.path.exists(downloaded_source_path):
                log.error("Download failed or source path not found after download. Aborting.")
                # Cleanup handled in finally block
                return False

            organization_success = self.organize(downloaded_source_path)

        except Exception as e:
             log.error(f"An unexpected error occurred during retrieve phase: {e}", exc_info=True)
             organization_success = False # Ensure failure state
        finally:
            self._cleanup() # Clean up temp download dir regardless

        if organization_success:
            log.info(f"Dataset retrieval process finished successfully for {self.dataset_name}.")
        else:
            log.error(f"Dataset organization failed for {self.dataset_name}.")
            # Clean up potentially incomplete final directory if organization failed
            if os.path.exists(self.dataset_final_path):
                 log.warning(f"Removing potentially incomplete dataset directory due to organization failure: {self.dataset_final_path}")
                 try:
                      shutil.rmtree(self.dataset_final_path)
                 except OSError as e:
                      log.error(f"Error removing incomplete directory {self.dataset_final_path}: {e}")

        return organization_success

    def _cleanup(self):
        """Removes the temporary download directory."""
        if os.path.exists(self.download_dir):
            log.info(f"Cleaning up temporary directory: {self.download_dir}")
            try:
                shutil.rmtree(self.download_dir)
            except OSError as e:
                log.error(f"Error removing temporary directory {self.download_dir}: {e}")