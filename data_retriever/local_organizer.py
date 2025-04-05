import os
import shutil
import logging
from .base_downloader import BaseDownloader
from .utils import extract_archive

log = logging.getLogger(__name__)

class LocalOrganizer(BaseDownloader):
    """Handles datasets provided via a local path, organizing them into the target structure."""
    def __init__(self, dataset_name, output_dir, download_dir, local_path, task_type):
        # Note: download_dir is not used for downloading here, but BaseDownloader needs it.
        # We pass a dummy path derived from output_dir to avoid conflicts.
        dummy_download_dir = download_dir or os.path.join(output_dir, dataset_name + "_localtmp")
        super().__init__(dataset_name, output_dir, dummy_download_dir)
        self.local_path = os.path.abspath(local_path)
        self.task_type = task_type

        # Ensure final path is different from local path to avoid recursive copy/move issues.
        if self.local_path == self.dataset_final_path:
             self.dataset_final_path = os.path.join(self.output_dir, self.dataset_name + "_organized")
             log.warning(f"Local path '{self.local_path}' matches target output path. Final organized data will be placed in: '{self.dataset_final_path}' to avoid conflicts.")


    def download(self):
        """'Download' step for local data: simply validates the path exists."""
        if not os.path.exists(self.local_path):
             log.error(f"Provided local path does not exist: {self.local_path}")
             return None
        log.info(f"Using existing local data at: {self.local_path}")
        return self.local_path # Return the validated local path as the 'source'

    def organize(self, source_path):
        """
        Organizes data from the local source_path into self.dataset_final_path.
        If source_path is a directory, copies its contents into 'all_data'.
        If source_path is an archive, extracts its contents into 'all_data'.
        Args:
            source_path (str): The validated local path (self.local_path).
        """
        log.info(f"Organizing from local path {source_path} into {self.dataset_final_path}")
        if source_path == self.dataset_final_path:
             # This case is handled by the __init__ adjustment, but double-check.
             log.warning("Source and destination paths are the same. Skipping organization step.")
             return True # Consider it 'organized' as it's already there.

        os.makedirs(self.dataset_final_path, exist_ok=True)
        # Generic copy/extract into 'all_data' - specific datasets might need overrides later
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        # Ensure target doesn't exist before copy/extract
        if os.path.exists(target_all_data):
             log.warning(f"Target 'all_data' directory exists. Removing before copy/extract: {target_all_data}")
             shutil.rmtree(target_all_data)
        os.makedirs(target_all_data) # Recreate empty target

        organized = False
        try:
            if os.path.isdir(source_path):
                # Copy contents of the directory
                log.debug(f"Copying directory contents from {source_path} to {target_all_data}")
                shutil.copytree(source_path, target_all_data, dirs_exist_ok=True)
                organized = True
            elif os.path.isfile(source_path):
                 # If local path is an archive file, extract it
                 log.debug(f"Attempting to extract archive file {source_path} to {target_all_data}")
                 extracted, _ = extract_archive(source_path, target_all_data, remove_archive=False) # Don't remove original local archive
                 if extracted:
                      log.info(f"Extracted local archive {source_path} to {target_all_data}")
                      organized = True
                 else:
                      # If not an archive, just copy the single file
                      log.warning(f"Local path {source_path} is a file but not a recognized archive. Copying file directly.")
                      shutil.copy2(source_path, target_all_data)
                      organized = True
            else:
                 log.error(f"Local source path {source_path} is neither a file nor a directory.")
                 organized = False


            if organized:
                 log.warning("Organization from local path used generic copy/extract into 'all_data'. Manual adjustment or specific organizers might be needed.")
                 # TODO: Add calls to specific standardization logic here if needed based on task_type/dataset_name
                 # Example: if self.dataset_name == 'cityscapes': self._standardize_cityscapes(self.dataset_final_path)
                 # Note: Standardization logic (like _standardize_structure in torchvision_downloader)
                 # would ideally be refactored into reusable functions or classes called here.
                 return True
            else:
                 log.warning(f"No data copied/extracted from local path: {source_path}")
                 return False
        except Exception as e:
            log.error(f"Error organizing local path '{source_path}': {e}", exc_info=True)
            return False

    def _cleanup(self):
        """No temporary download directory to clean for local organizer."""
        pass # Override base cleanup as it's not needed