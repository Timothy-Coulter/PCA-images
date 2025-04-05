import os
import argparse
import requests
import zipfile
import tarfile
import shutil
import logging
import importlib
from tqdm import tqdm
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_OUTPUT_DIR = "./datasets"
DEFAULT_DOWNLOAD_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "downloads_tmp")

# --- Helper Functions (moved from original script) ---

def download_file(url, destination_folder, filename=None):
    """Downloads a file from a URL with progress."""
    os.makedirs(destination_folder, exist_ok=True)
    local_filename = filename or url.split('/')[-1].split('?')[0]
    local_filepath = os.path.join(destination_folder, local_filename)

    try:
        with requests.get(url, stream=True, timeout=60) as r: # Increased timeout
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192

            logging.info(f"Downloading {local_filename} from {url}...")
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=local_filename, leave=False)
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                logging.error("ERROR, download size mismatch.")
                # Keep partial file for inspection? Or remove? For now, keep.
                # if os.path.exists(local_filepath): os.remove(local_filepath)
                # return None
            logging.info(f"Successfully downloaded {local_filename}")
            return local_filepath
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url}: {e}")
        if os.path.exists(local_filepath): os.remove(local_filepath)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during download: {e}")
        if os.path.exists(local_filepath): os.remove(local_filepath)
        return None

def extract_archive(filepath, destination_folder, remove_archive=True):
    """Extracts zip or tar archives."""
    if not os.path.exists(filepath):
        logging.error(f"Archive file not found: {filepath}")
        return False, None

    logging.info(f"Extracting {os.path.basename(filepath)} to {destination_folder}...")
    os.makedirs(destination_folder, exist_ok=True)
    extracted_path = destination_folder # Default if no single top-level folder

    try:
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                # Check for single top-level directory
                top_level_dirs = list(set(item.split('/')[0] for item in zip_ref.namelist()))
                if len(top_level_dirs) == 1:
                    extracted_path = os.path.join(destination_folder, top_level_dirs[0])
                zip_ref.extractall(destination_folder)
            logging.info("ZIP extraction complete.")
        elif tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r:*') as tar_ref:
                 # Check for single top-level directory
                members = tar_ref.getmembers()
                top_level_dirs = list(set(m.name.split('/')[0] for m in members if '/' in m.name))
                if len(top_level_dirs) == 1:
                     extracted_path = os.path.join(destination_folder, top_level_dirs[0])
                tar_ref.extractall(path=destination_folder)
            logging.info("TAR extraction complete.")
        else:
            logging.warning(f"File {os.path.basename(filepath)} is not a recognized archive format (zip/tar). Skipping extraction.")
            return False, filepath # Return original path if not archive

        if remove_archive:
            os.remove(filepath)
            logging.info(f"Removed archive file: {os.path.basename(filepath)}")
        return True, extracted_path # Indicate success and path to extracted content
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e: # Added EOFError
        logging.error(f"Error extracting archive {os.path.basename(filepath)}: {e}")
        return False, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during extraction: {e}")
        return False, None


# --- Base and Concrete Downloader Classes ---

class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders."""
    def __init__(self, dataset_name, output_dir, download_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.dataset_final_path = os.path.join(self.output_dir, self.dataset_name)

    @abstractmethod
    def download(self):
        """Downloads the raw dataset files. Returns path to downloaded content or None."""
        pass

    @abstractmethod
    def organize(self, source_path):
        """Organizes the downloaded data into the final structure. Returns True on success."""
        pass

    def retrieve(self):
        """Downloads and organizes the dataset."""
        if os.path.exists(self.dataset_final_path):
            logging.warning(f"Dataset directory '{self.dataset_final_path}' already exists. Skipping retrieval.")
            # Optionally add force flag to re-download/organize
            return True # Consider it success if already exists

        logging.info(f"Starting dataset retrieval for: {self.dataset_name}")
        logging.info(f"Target directory: {self.dataset_final_path}")
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        downloaded_source_path = self.download()

        if not downloaded_source_path or not os.path.exists(downloaded_source_path):
            logging.error("Download failed or source path not found. Aborting.")
            self._cleanup()
            return False

        success = self.organize(downloaded_source_path)

        self._cleanup() # Clean up temp download dir regardless of organization success

        if success:
            logging.info(f"Dataset retrieval process finished successfully for {self.dataset_name}.")
        else:
            logging.error(f"Dataset organization failed for {self.dataset_name}.")
            # Clean up potentially incomplete final directory?
            if os.path.exists(self.dataset_final_path):
                 logging.warning(f"Removing potentially incomplete dataset directory: {self.dataset_final_path}")
                 shutil.rmtree(self.dataset_final_path)

        return success

    def _cleanup(self):
        if os.path.exists(self.download_dir):
            logging.info(f"Cleaning up temporary directory: {self.download_dir}")
            try:
                shutil.rmtree(self.download_dir)
            except OSError as e:
                logging.error(f"Error removing temporary directory {self.download_dir}: {e}")


class UrlDownloader(BaseDownloader):
    """Downloads and extracts datasets from a URL."""
    def __init__(self, dataset_name, output_dir, download_dir, url, task_type, remove_archive=True):
        super().__init__(dataset_name, output_dir, download_dir)
        self.url = url
        self.task_type = task_type # Needed for generic organization
        self.remove_archive = remove_archive

    def download(self):
        downloaded_filepath = download_file(self.url, self.download_dir)
        if not downloaded_filepath:
            return None

        # Try to extract if it's an archive
        extract_dir = os.path.join(self.download_dir, "extracted_" + self.dataset_name)
        extracted, source_path = extract_archive(downloaded_filepath, extract_dir, remove_archive=self.remove_archive)

        # If extraction happened, source is the extracted content path.
        # If not an archive, source is the downloaded file itself.
        # If extraction failed, source_path is None.
        return source_path


    def organize(self, source_path):
        """Generic organization based on task type."""
        logging.info(f"Performing generic organization for task type '{self.task_type}' from {source_path}")
        if not source_path or not os.path.exists(source_path):
             logging.error(f"Source path '{source_path}' does not exist for organization.")
             return False

        os.makedirs(self.dataset_final_path, exist_ok=True)

        # Default: Copy everything into an 'all_data' subfolder in the final dataset directory
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        try:
            if os.path.isdir(source_path):
                # Copy contents of source_path into target_all_data
                for item in os.listdir(source_path):
                    s_item = os.path.join(source_path, item)
                    d_item = os.path.join(target_all_data, item)
                    if os.path.isdir(s_item):
                        shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                    else:
                        os.makedirs(target_all_data, exist_ok=True)
                        shutil.copy2(s_item, d_item)
                logging.info(f"Copied contents of {source_path} to {target_all_data}")
            elif os.path.isfile(source_path):
                 # If the source is just a single file (e.g., non-archive download)
                 os.makedirs(target_all_data, exist_ok=True)
                 shutil.copy2(source_path, target_all_data)
                 logging.info(f"Copied file {source_path} to {target_all_data}")
            else:
                 logging.error(f"Source path {source_path} is neither a file nor a directory.")
                 return False

            # Add specific logic hints based on task type if needed
            if self.task_type == 'classification':
                logging.warning("Generic organization used. For classification, expected structure is often dataset/split/class/image.jpg. Manual adjustment might be needed.")
            elif self.task_type == 'object_detection':
                 logging.warning("Generic organization used. For object detection, expected structure is often dataset/split/images/ and dataset/split/annotations/. Manual adjustment might be needed.")
            elif self.task_type == 'segmentation':
                 logging.warning("Generic organization used. For segmentation, expected structure is often dataset/split/images/ and dataset/split/masks/. Manual adjustment might be needed.")

            return True
        except Exception as e:
            logging.error(f"Error during generic organization: {e}")
            return False


class TorchvisionDownloader(BaseDownloader):
    """Uses torchvision.datasets to download and potentially organize."""
    def __init__(self, dataset_name, output_dir, download_dir, torchvision_dataset_name, task_type, **kwargs):
        super().__init__(dataset_name, output_dir, download_dir)
        self.torchvision_dataset_name = torchvision_dataset_name
        self.task_type = task_type
        self.kwargs = kwargs # e.g., split='train', download=True

        try:
            self.datasets = importlib.import_module('torchvision.datasets')
            self.dataset_class = getattr(self.datasets, self.torchvision_dataset_name)
        except ImportError:
            logging.error("torchvision not found. Please install it (`pip install torchvision`).")
            self.dataset_class = None
        except AttributeError:
            logging.error(f"Dataset '{self.torchvision_dataset_name}' not found in torchvision.datasets.")
            self.dataset_class = None

    def download(self):
        if not self.dataset_class:
            return None

        # Torchvision downloads directly to the specified root, often organizing itself.
        # We'll use the final dataset path as the root for torchvision.
        # Some datasets might require downloading multiple splits.
        logging.info(f"Using torchvision to download {self.torchvision_dataset_name} to {self.output_dir}...")

        try:
            # Handle datasets that need split specification vs those that download all
            if self.torchvision_dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
                 # These download both train/test if download=True
                 self.dataset_class(root=self.output_dir, train=True, download=True, **self.kwargs)
                 self.dataset_class(root=self.output_dir, train=False, download=True, **self.kwargs)
                 # The actual data might be inside a subfolder like 'cifar-10-batches-py'
                 # We need to find the actual root created by torchvision
                 # This part is tricky as torchvision internal paths can change.
                 # Let's assume the organization step will handle finding the data within self.output_dir
                 return self.output_dir # Return the base output dir, organization will sort it out

            elif self.torchvision_dataset_name in ['OxfordIIITPet', 'Flowers102', 'VOCSegmentation', 'VOCDetection', 'StanfordDogs']: # Added StanfordDogs
                 # These often require specifying splits or handle splits internally
                 splits = self.kwargs.get('split') # Check if a specific split is requested
                 download_kwargs = {'root': self.output_dir, 'download': True}
                 download_kwargs.update(self.kwargs) # Add other kwargs like year

                 if splits: # If specific split(s) are given in config
                     if not isinstance(splits, list): splits = [splits]
                     for split in splits:
                         try:
                             current_kwargs = download_kwargs.copy()
                             current_kwargs['split'] = split
                             self.dataset_class(**current_kwargs)
                         except TypeError as e:
                             if 'split' in str(e):
                                  logging.warning(f"Dataset {self.torchvision_dataset_name} might not support 'split' argument directly, attempting download without it.")
                                  self.dataset_class(**download_kwargs)
                                  break # Assume single download call is sufficient
                             else: raise e
                         except ValueError as e:
                              logging.warning(f"Split '{split}' might not be available for {self.torchvision_dataset_name}: {e}")
                 else:
                      # If no split is specified in config, try downloading without it
                      # Some datasets like OxfordIIITPet handle splits internally based on files
                      try:
                           self.dataset_class(**download_kwargs)
                      except TypeError as e:
                           # If it fails because split is required, log error
                           if 'required positional argument' in str(e) or 'required keyword-only argument' in str(e) and 'split' in str(e):
                                logging.error(f"Dataset {self.torchvision_dataset_name} requires a 'split' argument, but none was provided in config.")
                                return None
                           else: # Different error
                                raise e

                 return self.output_dir # Return base dir, organization needed

            else:
                 # Generic attempt for other datasets
                 self.dataset_class(root=self.output_dir, download=True, **self.kwargs)
                 return self.output_dir

        except Exception as e:
            logging.error(f"Error downloading with torchvision: {e}")
            return None

    def organize(self, source_path):
        # Torchvision often downloads into a structure like root/dataset_name/files...
        # We want the final structure to be root/OurDatasetName/split/...
        # This requires moving/renaming the folder created by torchvision.
        logging.info(f"Organizing torchvision dataset downloaded at {source_path}...")

        # Problem: torchvision download path is not always predictable (e.g., 'cifar-10-python').
        # We need to identify the folder torchvision created within source_path (self.output_dir).
        # This is fragile. A better approach might be to download to a temp dir first, then move.

        # Let's try a simplified approach: Assume torchvision created *some* folder directly
        # under self.output_dir that contains the data. We rename this folder to self.dataset_name.
        # This won't create the standard train/test/images/ structure, but keeps the data together.

        # Find potential candidate folders created by torchvision in self.output_dir
        candidates = [d for d in os.listdir(self.output_dir)
                      if os.path.isdir(os.path.join(self.output_dir, d)) and d != self.dataset_name and d != os.path.basename(DEFAULT_DOWNLOAD_DIR)]

        # Heuristic: Find a folder that seems related to the torchvision name
        torchvision_folder_name = None
        tv_name_lower = self.torchvision_dataset_name.lower()
        for cand in candidates:
             cand_lower = cand.lower().replace('-', '').replace('_','')
             # Simple check if candidate name contains parts of the torchvision name
             if tv_name_lower in cand_lower or cand_lower in tv_name_lower:
                  torchvision_folder_name = cand
                  break
        # Add specific known folder names
        if not torchvision_folder_name:
             known_folders = {
                 'CIFAR10': 'cifar-10-batches-py',
                 'CIFAR100': 'cifar-100-python',
                 'MNIST': 'MNIST',
                 'FashionMNIST': 'FashionMNIST',
                 'VOCDetection': 'VOCdevkit',
                 'VOCSegmentation': 'VOCdevkit',
                 'OxfordIIITPet': 'oxford-iiit-pet',
                 'Flowers102': 'flowers-102',
                 'StanfordDogs': 'stanford-dogs-dataset', # Check actual folder name created by torchvision
                 # Add more known mappings
             }
             if self.torchvision_dataset_name in known_folders and os.path.isdir(os.path.join(self.output_dir, known_folders[self.torchvision_dataset_name])):
                  torchvision_folder_name = known_folders[self.torchvision_dataset_name]


        if torchvision_folder_name:
            original_path = os.path.join(self.output_dir, torchvision_folder_name)
            logging.info(f"Found potential torchvision folder: {original_path}")
            if original_path != self.dataset_final_path:
                try:
                    logging.info(f"Renaming/Moving {original_path} to {self.dataset_final_path}")
                    # If target exists, remove it first? Or handle merge? Let's remove.
                    if os.path.exists(self.dataset_final_path):
                         logging.warning(f"Target path {self.dataset_final_path} exists, removing before move.")
                         shutil.rmtree(self.dataset_final_path)
                    shutil.move(original_path, self.dataset_final_path)
                    logging.info("Rename/Move successful.")
                    # Now, potentially standardize structure *within* self.dataset_final_path
                    self._standardize_structure(self.dataset_final_path)
                    return True
                except Exception as e:
                    logging.error(f"Failed to rename/move {original_path} to {self.dataset_final_path}: {e}")
                    # Attempt to copy if move failed?
                    return False
            else:
                 logging.info("Torchvision folder already matches target path. Attempting standardization.")
                 self._standardize_structure(self.dataset_final_path)
                 return True # Already correctly named

        else:
            # If the data was downloaded directly into self.output_dir without a specific subfolder
            # (less common, but possible), try standardizing there. Check if files exist.
            if any(f.lower().startswith(tv_name_lower[:5]) for f in os.listdir(self.output_dir)): # Heuristic check
                 logging.warning(f"Could not identify specific folder for {self.torchvision_dataset_name}. Attempting standardization directly in {self.output_dir}.")
                 # This is risky, might affect other datasets if output_dir is shared.
                 # A better approach: download to a unique temp dir first.
                 # For now, we proceed but with caution.
                 # We need to move the contents to the final path first.
                 temp_org_path = self.dataset_final_path + "_org_temp"
                 os.makedirs(temp_org_path, exist_ok=True)
                 moved_files = False
                 for item in os.listdir(self.output_dir):
                      # Avoid moving the target dir itself or the download dir
                      if item == self.dataset_name or item == os.path.basename(DEFAULT_DOWNLOAD_DIR): continue
                      try:
                           shutil.move(os.path.join(self.output_dir, item), os.path.join(temp_org_path, item))
                           moved_files = True
                      except Exception as e:
                           logging.error(f"Error moving item {item} during fallback organization: {e}")
                 if moved_files:
                      shutil.move(temp_org_path, self.dataset_final_path)
                      self._standardize_structure(self.dataset_final_path)
                      return True
                 else:
                      logging.error(f"Failed to move files for fallback organization of {self.torchvision_dataset_name}.")
                      return False

            else:
                 logging.error(f"Could not reliably identify the folder created by torchvision for {self.torchvision_dataset_name} in {self.output_dir}. Manual organization might be required.")
                 return False # Indicate organization might be incomplete

    def _standardize_structure(self, base_path):
        """Attempts to standardize common torchvision dataset structures."""
        logging.info(f"Attempting to standardize structure within {base_path} for {self.torchvision_dataset_name}...")

        # --- VOC Specific ---
        if self.torchvision_dataset_name in ['VOCDetection', 'VOCSegmentation']:
            voc_devkit_path = base_path # Assumes base_path is now the VOCdevkit folder
            voc_year_folder = None
            # Find the year folder (e.g., VOC2007, VOC2012)
            for item in os.listdir(voc_devkit_path):
                if item.startswith("VOC") and os.path.isdir(os.path.join(voc_devkit_path, item)):
                    voc_year_folder = os.path.join(voc_devkit_path, item)
                    break
            if not voc_year_folder:
                logging.warning("Could not find VOC year folder (e.g., VOC2007) inside VOCdevkit.")
                return

            # Create standard structure
            train_img_dir = os.path.join(base_path, 'train', 'images')
            train_ann_dir = os.path.join(base_path, 'train', 'annotations') # XML for detection, PNG for seg
            val_img_dir = os.path.join(base_path, 'val', 'images')
            val_ann_dir = os.path.join(base_path, 'val', 'annotations')
            # VOC often uses trainval/test splits defined in ImageSets
            imgset_folder = 'Segmentation' if self.task_type == 'segmentation' else 'Main'
            imgset_path = os.path.join(voc_year_folder, 'ImageSets', imgset_folder)

            seg_class_path = os.path.join(voc_year_folder, 'SegmentationClass')
            jpeg_path = os.path.join(voc_year_folder, 'JPEGImages')
            ann_xml_path = os.path.join(voc_year_folder, 'Annotations')

            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(train_ann_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(val_ann_dir, exist_ok=True)

            # Read train.txt and val.txt (or trainval.txt/test.txt)
            def process_split(split_name, target_img_dir, target_ann_dir):
                split_file = os.path.join(imgset_path, f"{split_name}.txt")
                if not os.path.exists(split_file):
                    # Fallback to Main if Segmentation split file doesn't exist
                    if imgset_folder == 'Segmentation':
                         split_file_main = os.path.join(voc_year_folder, 'ImageSets', 'Main', f"{split_name}.txt")
                         if os.path.exists(split_file_main):
                              logging.warning(f"Split file {split_file} not found, using {split_file_main} instead.")
                              split_file = split_file_main
                         else:
                              logging.warning(f"Split file not found: {split_file} (and not in Main)")
                              return
                    else:
                         logging.warning(f"Split file not found: {split_file}")
                         return

                with open(split_file, 'r') as f:
                    # Handle potential format differences (e.g., score presence)
                    image_ids = []
                    for line in f:
                         parts = line.strip().split()
                         if parts: image_ids.append(parts[0])

                for img_id in tqdm(image_ids, desc=f"Organizing VOC {split_name}", leave=False):
                    # Copy image
                    src_img = os.path.join(jpeg_path, f"{img_id}.jpg")
                    dst_img = os.path.join(target_img_dir, f"{img_id}.jpg")
                    if os.path.exists(src_img): shutil.copy2(src_img, dst_img)

                    # Copy annotation (XML for detection, PNG for segmentation)
                    if self.task_type == 'segmentation':
                        src_ann = os.path.join(seg_class_path, f"{img_id}.png")
                        dst_ann = os.path.join(target_ann_dir, f"{img_id}.png")
                    else: # Detection or other tasks using XML
                        src_ann = os.path.join(ann_xml_path, f"{img_id}.xml")
                        dst_ann = os.path.join(target_ann_dir, f"{img_id}.xml")

                    if os.path.exists(src_ann): shutil.copy2(src_ann, dst_ann)

            # Determine which splits to process based on torchvision download args if possible
            requested_split = self.kwargs.get('image_set', 'trainval') # Default to trainval if not specified
            if requested_split == 'train':
                 process_split('train', train_img_dir, train_ann_dir)
            elif requested_split == 'val':
                 process_split('val', val_img_dir, val_ann_dir)
            elif requested_split == 'test':
                 # Test set often lacks annotations
                 test_img_dir = os.path.join(base_path, 'test', 'images')
                 os.makedirs(test_img_dir, exist_ok=True)
                 process_split('test', test_img_dir, os.path.join(base_path, 'test', 'annotations')) # Annot dir might remain empty
            elif requested_split == 'trainval':
                 process_split('train', train_img_dir, train_ann_dir)
                 process_split('val', train_img_dir, train_ann_dir) # Combine train and val into train split
                 # Or should we keep train/val separate? Let's combine for simplicity now.
                 # process_split('val', val_img_dir, val_ann_dir) # Keep separate

            logging.info(f"VOC dataset standardized structure created in {base_path}")
            # Optionally remove original VOCdevkit structure if desired

        # --- OxfordPets / Flowers102 / StanfordDogs ---
        elif self.torchvision_dataset_name in ['OxfordIIITPet', 'Flowers102', 'StanfordDogs']:
             # These often download into 'images' and 'annotations' folders already
             # StanfordDogs might just have 'images' and separate annotation files/lists
             images_dir = os.path.join(base_path, 'images')
             annotations_dir = os.path.join(base_path, 'annotations') # Pets: trimaps, Flowers: N/A?, Dogs: XMLs?
             lists_dir = os.path.join(base_path, 'lists') # Stanford Dogs uses lists for splits

             if os.path.isdir(images_dir):
                  logging.info(f"Found 'images' dir in {base_path}. Assuming single split or needs list-based splitting.")
                  # Create a default 'all_data' structure initially
                  all_img_dir = os.path.join(base_path, 'all_data', 'images')
                  all_ann_dir = os.path.join(base_path, 'all_data', ('annotations' if os.path.isdir(annotations_dir) else 'masks'))

                  os.makedirs(all_img_dir, exist_ok=True)
                  # Move contents of images_dir to all_img_dir
                  for item in os.listdir(images_dir):
                       shutil.move(os.path.join(images_dir, item), os.path.join(all_img_dir, item))
                  os.rmdir(images_dir) # Remove now empty images dir

                  if os.path.isdir(annotations_dir):
                       os.makedirs(all_ann_dir, exist_ok=True)
                       for item in os.listdir(annotations_dir):
                            shutil.move(os.path.join(annotations_dir, item), os.path.join(all_ann_dir, item))
                       os.rmdir(annotations_dir) # Remove now empty annotations dir

                  logging.info(f"Moved {self.torchvision_dataset_name} data to 'all_data' structure.")

                  # --- Stanford Dogs Specific Split ---
                  if self.torchvision_dataset_name == 'StanfordDogs' and os.path.isdir(lists_dir):
                       logging.info("Applying Stanford Dogs train/test split based on lists...")
                       train_img_dir = os.path.join(base_path, 'train', 'images')
                       test_img_dir = os.path.join(base_path, 'test', 'images')
                       # Annotations are often separate XML files per image, need to find them
                       dog_ann_dir = os.path.join(base_path, 'Annotation') # Check actual name

                       os.makedirs(train_img_dir, exist_ok=True)
                       os.makedirs(test_img_dir, exist_ok=True)
                       train_ann_dir = os.path.join(base_path, 'train', 'annotations')
                       test_ann_dir = os.path.join(base_path, 'test', 'annotations')
                       if os.path.isdir(dog_ann_dir):
                            os.makedirs(train_ann_dir, exist_ok=True)
                            os.makedirs(test_ann_dir, exist_ok=True)

                       def process_dog_split(list_file, target_img_dir, target_ann_dir):
                           if not os.path.exists(list_file):
                                logging.warning(f"Dog split file not found: {list_file}")
                                return
                           with open(list_file, 'r') as f:
                                image_files = [line.strip() for line in f]
                           for img_rel_path in tqdm(image_files, desc=f"Organizing Dogs {os.path.basename(list_file)}", leave=False):
                               src_img = os.path.join(all_img_dir, img_rel_path)
                               dst_img = os.path.join(target_img_dir, img_rel_path)
                               if os.path.exists(src_img):
                                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                                    shutil.move(src_img, dst_img) # Move from all_data

                               # Move corresponding annotation if exists
                               if os.path.isdir(dog_ann_dir):
                                    ann_rel_path = os.path.splitext(img_rel_path)[0] # Assumes XML in Annotation/breed/image_id
                                    src_ann = os.path.join(dog_ann_dir, ann_rel_path) # Path needs verification
                                    dst_ann = os.path.join(target_ann_dir, ann_rel_path)
                                    # This assumes annotation structure matches image structure - needs check
                                    # A better way might be to find XMLs based on image ID directly
                                    potential_ann_folder = os.path.join(dog_ann_dir, img_rel_path.split('/')[0]) # e.g., Annotation/n02085620-Chihuahua/
                                    img_id = os.path.splitext(os.path.basename(img_rel_path))[0]
                                    src_ann_alt = os.path.join(potential_ann_folder, img_id) # No extension needed? Check format.

                                    if os.path.exists(src_ann_alt): # Check alternative path
                                         os.makedirs(os.path.dirname(dst_ann), exist_ok=True)
                                         shutil.move(src_ann_alt, dst_ann) # Move from Annotation dir

                       process_dog_split(os.path.join(lists_dir, 'train_list.txt'), train_img_dir, train_ann_dir)
                       process_dog_split(os.path.join(lists_dir, 'test_list.txt'), test_img_dir, test_ann_dir)

                       # Remove original Annotation and lists dir? And empty all_data?
                       if os.path.isdir(dog_ann_dir): shutil.rmtree(dog_ann_dir)
                       shutil.rmtree(lists_dir)
                       try: os.rmdir(all_img_dir) # Remove if empty
                       except OSError: pass
                       try: os.rmdir(os.path.dirname(all_img_dir)) # Remove all_data if empty
                       except OSError: pass
                       logging.info("Stanford Dogs train/test split applied.")

             else:
                  logging.warning(f"Could not find expected 'images' directory in {base_path} for {self.torchvision_dataset_name}. Standardization skipped.")


        # --- CIFAR / MNIST --- (Stored in pickled/binary files, less about folder structure)
        elif self.torchvision_dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
             # The data is usually in specific files (e.g., data_batch_1, training.pt).
             # No standard folder reorganization is typically needed here.
             # Visualizer will need to know how to load from these files.
             logging.info(f"{self.torchvision_dataset_name} uses specific file formats. No folder standardization applied.")
             pass # Nothing to standardize in terms of folders usually

        else:
             logging.info(f"No specific standardization logic implemented for {self.torchvision_dataset_name}. Using downloaded structure.")


class KaggleDownloader(BaseDownloader):
    """Downloads datasets from Kaggle using the Kaggle API."""
    def __init__(self, dataset_name, output_dir, download_dir, kaggle_id, task_type):
        super().__init__(dataset_name, output_dir, download_dir)
        self.kaggle_id = kaggle_id
        self.task_type = task_type
        self.kaggle_api = self._setup_kaggle_api()

    def _setup_kaggle_api(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            logging.info("Kaggle API authenticated successfully.")
            return api
        except ImportError:
            logging.error("Kaggle API client not installed. Please run 'pip install kaggle'.")
            return None
        except Exception as e:
            logging.error(f"Kaggle API authentication failed. Ensure kaggle.json is configured correctly: {e}")
            return None

    def download(self):
        if not self.kaggle_api:
            return None

        logging.info(f"Downloading dataset '{self.kaggle_id}' from Kaggle to {self.download_dir}...")
        try:
            # kaggle datasets download [-f <file_name>] -d <dataset_id> -p <path> --unzip
            self.kaggle_api.dataset_download_files(self.kaggle_id, path=self.download_dir, unzip=True, quiet=False)
            logging.info(f"Kaggle dataset '{self.kaggle_id}' downloaded and extracted.")
            # The files are extracted directly into download_dir.
            return self.download_dir # Source for organization is the download dir itself
        except Exception as e:
            logging.error(f"Error downloading dataset from Kaggle: {e}")
            return None

    def organize(self, source_path):
        """Generic organization for Kaggle datasets."""
        # Kaggle datasets vary wildly in structure. Apply generic copy.
        logging.info(f"Performing generic organization for Kaggle dataset from {source_path}")
        if not source_path or not os.path.isdir(source_path):
             logging.error(f"Source path '{source_path}' is not a valid directory for organization.")
             return False

        os.makedirs(self.dataset_final_path, exist_ok=True)
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        os.makedirs(target_all_data, exist_ok=True)

        try:
            # Copy contents of source_path (download_dir) into target_all_data
            for item in os.listdir(source_path):
                s_item = os.path.join(source_path, item)
                # Avoid copying the final dataset dir into itself if output is subdir of download
                if os.path.abspath(s_item) == os.path.abspath(self.dataset_final_path):
                    continue
                d_item = os.path.join(target_all_data, item)
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(s_item, d_item)
            logging.info(f"Copied contents of {source_path} to {target_all_data}")
            logging.warning("Generic organization used for Kaggle dataset. Manual adjustment might be needed based on specific dataset structure.")
            return True
        except Exception as e:
            logging.error(f"Error during generic organization for Kaggle dataset: {e}")
            return False


# --- Dataset Registry and Factory ---

DATASET_CONFIG = {
    # Small datasets
    "cifar10": {"type": "torchvision", "id": "CIFAR10", "task": "classification"},
    "cifar100": {"type": "torchvision", "id": "CIFAR100", "task": "classification"},
    "mnist": {"type": "torchvision", "id": "MNIST", "task": "classification"},
    "fashion_mnist": {"type": "torchvision", "id": "FashionMNIST", "task": "classification"}, # Added
    "voc2007_detect": {"type": "torchvision", "id": "VOCDetection", "task": "object_detection", "kwargs": {"year": "2007", "image_set": "trainval"}}, # Specify year/split
    "voc2007_segment": {"type": "torchvision", "id": "VOCSegmentation", "task": "segmentation", "kwargs": {"year": "2007", "image_set": "trainval"}},
    # Note: VOC test sets often don't have public annotations
    "camvid": {"type": "url", "id": "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip", "task": "segmentation", "note": "Check URL. Structure needs specific organizer."}, # Example URL, check official source
    # OpenImages subset - Very complex, requires dedicated tools or pre-made subset URL
    "openimages_subset": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download or specific tools (e.g., FiftyOne). Provide path via --local-path."},

    # Medium datasets
    "oxford_pets": {"type": "torchvision", "id": "OxfordIIITPet", "task": "segmentation"}, # Also classification
    "flowers102": {"type": "torchvision", "id": "Flowers102", "task": "classification"},
    "stanford_dogs": {"type": "torchvision", "id": "StanfordDogs", "task": "classification"}, # Torchvision handles download/extraction
    # COCO - Complex, multiple files (images, annotations)
    "coco2017_detect": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download of images (train/val) and annotations (train/val). Provide path via --local-path."},
    "coco2017_segment": {"type": "manual", "id": None, "task": "segmentation", "note": "Requires manual download of images (train/val) and annotations (train/val). Provide path via --local-path."},
    # KITTI - Complex, multiple tasks/formats
    "kitti_detect": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download from KITTI website. Provide path via --local-path."},
    # ADE20K - Large segmentation dataset
    "ade20k": {"type": "url", "id": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip", "task": "segmentation", "note": "Check URL. Structure needs specific organizer."}, # Check official source/format
    # Cityscapes - Requires registration
    "cityscapes": {"type": "manual", "id": None, "task": "segmentation", "note": "Requires registration and manual download from Cityscapes website (gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip). Provide path via --local-path."},
}


class LocalOrganizer(BaseDownloader):
    """Handles datasets provided via a local path."""
    def __init__(self, dataset_name, output_dir, local_path, task_type):
        # Note: download_dir is not used here, but needed for super init
        super().__init__(dataset_name, output_dir, os.path.join(output_dir, dataset_name + "_localtmp"))
        self.local_path = os.path.abspath(local_path)
        self.task_type = task_type
        # Ensure final path is different from local path to avoid recursive copy
        if self.local_path == self.dataset_final_path:
             self.dataset_final_path = os.path.join(self.output_dir, self.dataset_name + "_organized")
             logging.warning(f"Local path matches output path. Final organized data will be in: {self.dataset_final_path}")


    def download(self):
        # 'Download' is just returning the existing local path
        if not os.path.exists(self.local_path):
             logging.error(f"Provided local path does not exist: {self.local_path}")
             return None
        return self.local_path

    def organize(self, source_path):
        # source_path is self.local_path here
        logging.info(f"Organizing from local path {source_path} into {self.dataset_final_path}")
        if source_path == self.dataset_final_path:
             logging.warning("Source and destination paths are the same after adjustment. Skipping organization.")
             # Or should we attempt to organize in-place? Risky.
             return True # Consider it 'organized' as it's already there.

        os.makedirs(self.dataset_final_path, exist_ok=True)
        # Generic copy into 'all_data' - specific datasets might need overrides
        target_all_data = os.path.join(self.dataset_final_path, 'all_data')
        os.makedirs(target_all_data, exist_ok=True)
        copied = False
        try:
            if os.path.isdir(source_path):
                for item in os.listdir(source_path):
                    s_item = os.path.join(source_path, item)
                    # Avoid copying the target dir if it somehow ended up inside source
                    if os.path.abspath(s_item) == os.path.abspath(self.dataset_final_path): continue
                    d_item = os.path.join(target_all_data, item)
                    if os.path.isdir(s_item):
                        shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s_item, d_item)
                    copied = True
            elif os.path.isfile(source_path): # If local path is an archive file
                 extracted, extracted_path = extract_archive(source_path, target_all_data, remove_archive=False)
                 if extracted:
                      logging.info(f"Extracted local archive {source_path} to {target_all_data}")
                      # If extraction created a single subfolder, move its contents up? Optional.
                      copied = True
                 else:
                      logging.warning(f"Local path {source_path} is a file but not a recognized archive. Copying directly.")
                      shutil.copy2(source_path, target_all_data)
                      copied = True

            if copied:
                 logging.warning("Organization from local path used generic copy/extract. Manual adjustment or specific organizers might be needed.")
                 # TODO: Add calls to specific standardization logic here if needed based on task_type/dataset_name
                 # Example: if self.dataset_name == 'cityscapes': self._standardize_cityscapes(self.dataset_final_path)
                 return True
            else:
                 logging.warning(f"No data copied/extracted from local path: {source_path}")
                 return False
        except Exception as e:
            logging.error(f"Error organizing local path: {e}")
            return False

    def _cleanup(self): pass # No temp download dir to clean


def get_downloader(dataset_key, output_dir=DEFAULT_OUTPUT_DIR, download_dir=DEFAULT_DOWNLOAD_DIR, local_path=None):
    """Factory function to get the appropriate downloader instance."""
    if dataset_key not in DATASET_CONFIG:
        # Allow generic URL/Kaggle if key not found? No, stick to defined keys or explicit URL/Kaggle args.
        logging.error(f"Unknown dataset key: {dataset_key}. Available keys: {list(DATASET_CONFIG.keys())}")
        return None

    config = DATASET_CONFIG[dataset_key]
    dataset_name = dataset_key # Use the key as the folder name
    dataset_type = config["type"]
    dataset_id = config["id"]
    task_type = config["task"]
    kwargs = config.get("kwargs", {})

    # Handle local path override or manual datasets
    if local_path:
         local_path = os.path.abspath(local_path)
         if dataset_type == "manual":
              if not os.path.exists(local_path):
                   logging.error(f"Dataset '{dataset_key}' is manual, but local path '{local_path}' not found. Please download manually.")
                   return None
              else:
                   logging.info(f"Using local path '{local_path}' for manual dataset '{dataset_key}'.")
                   return LocalOrganizer(dataset_name, output_dir, local_path, task_type)
         else:
              # If local path exists for a non-manual dataset, use LocalOrganizer instead of downloading
              if os.path.exists(local_path):
                   logging.warning(f"Local path '{local_path}' provided and exists for non-manual dataset '{dataset_key}'. Using local data instead of downloading.")
                   return LocalOrganizer(dataset_name, output_dir, local_path, task_type)
              else:
                   logging.warning(f"Local path '{local_path}' provided but does not exist. Proceeding with configured download type '{dataset_type}'.")

    # Proceed with configured download type if local path wasn't used
    if dataset_type == "torchvision":
        return TorchvisionDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type, **kwargs)
    elif dataset_type == "url":
        # TODO: Add specific organizers for CamVid, ADE20k based on their structure after download
        logging.warning(f"Using generic URL downloader for {dataset_key}. Specific organization logic might be missing.")
        return UrlDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type)
    elif dataset_type == "kaggle":
        return KaggleDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type)
    elif dataset_type == "manual":
         # This case should only be reached if local_path was None or didn't exist
         logging.error(f"Dataset '{dataset_key}' requires manual download. Please download it and provide the path using --local-path.")
         return None
    else:
        logging.error(f"Unsupported dataset type '{dataset_type}' in config for '{dataset_key}'.")
        return None


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize image datasets using predefined configurations or URLs/Kaggle IDs.")

    # Option 1: Use predefined dataset key
    parser.add_argument("-d", "--dataset-key", type=str, # choices=list(DATASET_CONFIG.keys()), # Allow any key for potential future additions
                        help="Key of the predefined dataset to download (e.g., 'cifar10', 'voc2007_detect'). See DATASET_CONFIG in script for available keys.")

    # Option 2: Specify URL manually (acts as a generic downloader if key not found/used)
    parser.add_argument("-u", "--url", type=str, help="URL of the dataset file (zip, tar.gz, etc.) for generic download. Overrides --dataset-key if provided.")
    parser.add_argument("--url-task-type", type=str, default="other",
                        choices=['classification', 'object_detection', 'segmentation', 'unlabelled', 'other'],
                        help="Task type for generic URL download organization (default: other).")
    parser.add_argument("--url-dataset-name", type=str, help="Name for the dataset folder when using --url (default: derived from URL).")


    # Option 3: Specify Kaggle ID manually (acts as a generic downloader)
    parser.add_argument("-k", "--kaggle-id", type=str, help="Kaggle dataset identifier (e.g., 'user/dataset-name'). Requires Kaggle API setup. Overrides --dataset-key and --url.")
    parser.add_argument("--kaggle-task-type", type=str, default="other",
                        choices=['classification', 'object_detection', 'segmentation', 'unlabelled', 'other'],
                        help="Task type for generic Kaggle download organization (default: other).")
    parser.add_argument("--kaggle-dataset-name", type=str, help="Name for the dataset folder when using --kaggle-id (default: derived from ID).")

    # Option 4: Specify local path (for manual downloads or existing datasets)
    parser.add_argument("-l", "--local-path", type=str, help="Path to an already downloaded dataset folder or archive. If provided with a --dataset-key of type 'manual', this path MUST be used. If provided with other types, this path will be used INSTEAD of downloading if it exists.")


    # General options
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Base directory to store the final dataset folders (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--download-dir", type=str, # Default computed dynamically
                        help=f"Temporary directory for downloads (default: <output_dir>/downloads_tmp).")

    args = parser.parse_args()

    # Determine downloader based on input priority: Local Path (if exists) > URL > Kaggle > Dataset Key
    downloader = None
    final_output_dir = os.path.abspath(args.output_dir)
    final_download_dir = os.path.abspath(args.download_dir or os.path.join(final_output_dir, "downloads_tmp")) # Default download dir relative to output dir

    # --- Input Source Resolution ---
    dataset_source_specified = False
    if args.local_path:
         # Local path takes precedence if it exists, regardless of other flags (except for manual keys where it's required)
         abs_local_path = os.path.abspath(args.local_path)
         if os.path.exists(abs_local_path):
              dataset_key_for_local = args.dataset_key # Use provided key if available for naming/task type
              if dataset_key_for_local and dataset_key_for_local in DATASET_CONFIG:
                   config = DATASET_CONFIG[dataset_key_for_local]
                   dataset_name = dataset_key_for_local
                   task_type = config['task']
                   logging.info(f"Using local path '{abs_local_path}' with config from key '{dataset_key_for_local}'.")
              else:
                   dataset_name = os.path.basename(abs_local_path).split('.')[0] # Basic name from path
                   task_type = 'other' # Unknown task type
                   logging.info(f"Using local path '{abs_local_path}'. Dataset key not provided or invalid, using generic organization.")
              downloader = LocalOrganizer(dataset_name, final_output_dir, abs_local_path, task_type)
              dataset_source_specified = True
         elif args.dataset_key and args.dataset_key in DATASET_CONFIG and DATASET_CONFIG[args.dataset_key]['type'] == 'manual':
              # If it's a manual key and local path doesn't exist, it's an error
              logging.error(f"Dataset '{args.dataset_key}' is manual, but required --local-path '{abs_local_path}' not found.")
              parser.error(f"Local path '{args.local_path}' not found for manual dataset '{args.dataset_key}'.")
         else:
              # Local path provided but doesn't exist, and not for a manual key. Ignore it and proceed.
              logging.warning(f"Provided --local-path '{args.local_path}' not found. Ignoring and attempting download based on other flags.")

    if not dataset_source_specified:
        if args.url:
            dataset_name = args.url_dataset_name or args.url.split('/')[-1].split('.')[0]
            logging.info(f"Using generic URL downloader for {args.url}")
            downloader = UrlDownloader(dataset_name, final_output_dir, final_download_dir, args.url, args.url_task_type)
            dataset_source_specified = True
        elif args.kaggle_id:
             dataset_name = args.kaggle_dataset_name or args.kaggle_id.split('/')[-1]
             logging.info(f"Using Kaggle downloader for {args.kaggle_id}")
             downloader = KaggleDownloader(dataset_name, final_output_dir, final_download_dir, args.kaggle_id, args.kaggle_task_type)
             dataset_source_specified = True
        elif args.dataset_key:
            logging.info(f"Using predefined dataset configuration for key: {args.dataset_key}")
            # Pass local_path=None here, as the check above determined it shouldn't be used for download override
            downloader = get_downloader(args.dataset_key, final_output_dir, final_download_dir, local_path=None)
            if downloader: dataset_source_specified = True
        # else: # No source specified if we reach here (and local path didn't exist)
             # Error handled below

    # --- Final Check and Execution ---
    if not dataset_source_specified:
         parser.error("No valid dataset source specified or found. Please use --dataset-key, --url, --kaggle-id, or provide a valid --local-path.")

    if downloader:
        success = downloader.retrieve()
        if success:
            logging.info(f"Dataset retrieval process completed for '{downloader.dataset_name}'. Final location: {downloader.dataset_final_path}")
            print(f"\nDataset ready at: {downloader.dataset_final_path}") # Print final path clearly
        else:
            logging.error(f"Dataset retrieval process failed for '{downloader.dataset_name}'.")
            exit(1)
    else:
        # Error should have been logged by get_downloader or the logic above
        logging.error("Could not initialize a suitable downloader.")
        exit(1)
