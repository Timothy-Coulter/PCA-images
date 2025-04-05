import os
import shutil
import logging
from .base_downloader import BaseDownloader
from .utils import download_file, extract_archive

log = logging.getLogger(__name__)

class UrlDownloader(BaseDownloader):
    """Downloads and extracts datasets from a URL."""
    def __init__(self, dataset_name, output_dir, download_dir, url, task_type, remove_archive=True):
        super().__init__(dataset_name, output_dir, download_dir)
        self.url = url
        self.task_type = task_type # Needed for generic organization hints
        self.remove_archive = remove_archive

    def download(self):
        """Downloads the file from the URL."""
        downloaded_filepath = download_file(self.url, self.download_dir)
        if not downloaded_filepath:
            return None

        # Try to extract if it's an archive, otherwise return the downloaded file path
        extract_dir = os.path.join(self.download_dir, "extracted_" + self.dataset_name)
        extracted, source_path = extract_archive(downloaded_filepath, extract_dir, remove_archive=self.remove_archive)

        # If extraction happened, source is the extracted content path.
        # If not an archive, source is the downloaded file itself.
        # If extraction failed, source_path is None.
        return source_path


    def organize(self, source_path):
        """
        Generic organization: copies the contents of source_path into
        'all_data' subdirectory within the final dataset directory.
        """
        log.info(f"Performing generic organization for task type '{self.task_type}' from {source_path}")
        if not source_path or not os.path.exists(source_path):
             log.error(f"Source path '{source_path}' does not exist for organization.")
             return False

        os.makedirs(self.dataset_final_path, exist_ok=True)

        # Default: Copy everything into an 'all_data' subfolder in the final dataset directory
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        try:
            if os.path.isdir(source_path):
                # Copy contents of source_path into target_all_data
                # Use copytree for simplicity if source is a directory
                shutil.copytree(source_path, target_all_data, dirs_exist_ok=True)
                log.info(f"Copied contents of directory {source_path} to {target_all_data}")
            elif os.path.isfile(source_path):
                 # If the source is just a single file (e.g., non-archive download)
                 os.makedirs(target_all_data, exist_ok=True)
                 shutil.copy2(source_path, target_all_data)
                 log.info(f"Copied file {source_path} to {target_all_data}")
            else:
                 log.error(f"Source path {source_path} is neither a file nor a directory.")
                 return False

            # Add specific logic hints based on task type if needed
            if self.task_type == 'classification':
                log.warning("Generic organization used. For classification, expected structure is often dataset/split/class/image.jpg. Manual adjustment might be needed.")
            elif self.task_type == 'object_detection':
                 log.warning("Generic organization used. For object detection, expected structure is often dataset/split/images/ and dataset/split/annotations/. Manual adjustment might be needed.")
            elif self.task_type == 'segmentation':
                 log.warning("Generic organization used. For segmentation, expected structure is often dataset/split/images/ and dataset/split/masks/. Manual adjustment might be needed.")

            # TODO: Implement specific organizers for datasets downloaded via URL (e.g., CamVid, ADE20k)
            # These could be separate classes inheriting from UrlDownloader or called here based on dataset_name.
            # Example:
            # if self.dataset_name == 'camvid':
            #     return self._organize_camvid(target_all_data)

            return True
        except Exception as e:
            log.error(f"Error during generic organization: {e}", exc_info=True)
            return False

    # Example of a specific organizer method (needs implementation)
    # def _organize_camvid(self, data_path):
    #     log.info(f"Running specific organization for CamVid in {data_path}...")
    #     # Logic to create train/val/test splits with images/masks folders
    #     return True