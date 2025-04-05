import os
import shutil
import logging
import importlib
from tqdm import tqdm
from .base_downloader import BaseDownloader

log = logging.getLogger(__name__)

# Known folder names created by torchvision downloads (might need updates)
KNOWN_TORCHVISION_FOLDERS = {
    'CIFAR10': 'cifar-10-batches-py',
    'CIFAR100': 'cifar-100-python',
    'MNIST': 'MNIST',
    'FashionMNIST': 'FashionMNIST',
    'VOCDetection': 'VOCdevkit',
    'VOCSegmentation': 'VOCdevkit',
    'OxfordIIITPet': 'oxford-iiit-pet',
    'Flowers102': 'flowers-102',
    'StanfordDogs': 'stanford-dogs-dataset', # Verify this exact name
    'Cityscapes': 'cityscapes', # Verify this exact name
    # Add more known mappings as discovered
}


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
            log.debug(f"Successfully imported torchvision.datasets.{self.torchvision_dataset_name}")
        except ImportError:
            log.error("torchvision not found. Please install it (`pip install torchvision`).")
            self.dataset_class = None
        except AttributeError:
            log.error(f"Dataset '{self.torchvision_dataset_name}' not found in torchvision.datasets.")
            self.dataset_class = None
        except Exception as e:
             log.error(f"Unexpected error importing torchvision dataset {self.torchvision_dataset_name}: {e}")
             self.dataset_class = None

    def download(self):
        """Downloads data using the specified torchvision dataset class."""
        if not self.dataset_class:
            return None

        # --- Download Strategy ---
        # Torchvision usually downloads to root/dataset_name/.
        # To avoid clutter and potential conflicts in the main output_dir,
        # we'll download to a temporary location *within* the main download_dir,
        # specific to this dataset instance.
        temp_torchvision_root = os.path.join(self.download_dir, f"torchvision_{self.dataset_name}_temp")
        os.makedirs(temp_torchvision_root, exist_ok=True)
        log.info(f"Using torchvision to download {self.torchvision_dataset_name} to temporary root: {temp_torchvision_root}...")

        try:
            # Prepare kwargs for the dataset constructor
            download_kwargs = {'root': temp_torchvision_root, 'download': True}
            # Filter out non-standard kwargs if necessary, or pass all (**self.kwargs)
            # For simplicity, pass all kwargs from config for now.
            download_kwargs.update(self.kwargs)

            # Handle datasets that need split specification vs those that download all
            # This logic might need refinement based on specific dataset behaviors in torchvision
            if self.torchvision_dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
                 # These download both train/test if download=True and train=True/False is called
                 log.debug(f"Attempting train split download for {self.torchvision_dataset_name}")
                 self.dataset_class(train=True, **download_kwargs)
                 log.debug(f"Attempting test split download for {self.torchvision_dataset_name}")
                 self.dataset_class(train=False, **download_kwargs)

            elif self.torchvision_dataset_name in ['OxfordIIITPet', 'Flowers102', 'VOCSegmentation', 'VOCDetection', 'StanfordDogs', 'Cityscapes']:
                 # These often require specifying splits or handle splits internally
                 splits_to_download = download_kwargs.pop('split', None) # Get split from kwargs if present
                 image_set = download_kwargs.pop('image_set', None) # For VOC

                 # Determine splits based on common practices or kwargs
                 if not splits_to_download:
                      if self.torchvision_dataset_name in ['VOCSegmentation', 'VOCDetection']:
                           splits_to_download = image_set or 'trainval' # Default VOC split
                      elif self.torchvision_dataset_name == 'OxfordIIITPet':
                           splits_to_download = ['trainval', 'test'] # Default Pet splits
                      elif self.torchvision_dataset_name == 'Flowers102':
                           splits_to_download = ['train', 'val', 'test'] # Default Flowers splits
                      elif self.torchvision_dataset_name == 'StanfordDogs':
                           splits_to_download = ['train', 'test'] # Default Dogs splits
                      elif self.torchvision_dataset_name == 'Cityscapes':
                           splits_to_download = ['train', 'val', 'test'] # Default Cityscapes splits
                           # Cityscapes also needs mode ('fine', 'coarse') and target_type
                           if 'mode' not in download_kwargs: download_kwargs['mode'] = 'fine'
                           if 'target_type' not in download_kwargs: download_kwargs['target_type'] = 'semantic'


                 if not isinstance(splits_to_download, list): splits_to_download = [splits_to_download]

                 log.debug(f"Attempting to download splits {splits_to_download} for {self.torchvision_dataset_name}")
                 for split in splits_to_download:
                     current_kwargs = download_kwargs.copy()
                     # Add split/image_set back depending on dataset requirements
                     if self.torchvision_dataset_name in ['VOCSegmentation', 'VOCDetection']:
                          current_kwargs['image_set'] = split
                     else:
                          current_kwargs['split'] = split

                     try:
                         log.debug(f"Downloading split '{split}' with kwargs: {current_kwargs}")
                         self.dataset_class(**current_kwargs)
                     except TypeError as e:
                         # Handle cases where 'split' or 'image_set' is not expected
                         if 'split' in str(e) or 'image_set' in str(e):
                              log.warning(f"Dataset {self.torchvision_dataset_name} might not support split/image_set argument '{split}' directly, attempting download without it.")
                              # Try downloading once without split/image_set
                              base_kwargs = download_kwargs.copy()
                              if 'split' in base_kwargs: del base_kwargs['split']
                              if 'image_set' in base_kwargs: del base_kwargs['image_set']
                              try:
                                   self.dataset_class(**base_kwargs)
                                   log.debug("Single download call successful.")
                                   break # Assume single download call is sufficient
                              except Exception as inner_e:
                                   log.error(f"Single download call also failed: {inner_e}")
                                   return None # Failed to download
                         else: raise e # Different TypeError
                     except ValueError as e:
                          # Handle cases where a specific split is not available
                          log.warning(f"Split '{split}' might not be available or valid for {self.torchvision_dataset_name}: {e}")
                     except RuntimeError as e:
                          # Handle common errors like dataset not found locally after failed download
                          log.error(f"Runtime error during download for split '{split}': {e}")
                          # Continue trying other splits? Or fail? Let's fail for now.
                          return None


            else:
                 # Generic attempt for other datasets not explicitly handled
                 log.debug(f"Attempting generic download for {self.torchvision_dataset_name}")
                 self.dataset_class(**download_kwargs)

            log.info(f"Torchvision download completed in {temp_torchvision_root}")
            # The actual data might be inside a subfolder (e.g., cifar-10-batches-py) within temp_torchvision_root
            return temp_torchvision_root # Return the temp root, organization will find the actual data folder

        except Exception as e:
            log.error(f"Error downloading {self.torchvision_dataset_name} with torchvision: {e}", exc_info=True)
            return None

    def organize(self, source_path):
        """
        Organizes the torchvision dataset by moving/renaming the downloaded folder
        and potentially standardizing its internal structure.
        Args:
            source_path (str): The temporary root directory where torchvision downloaded data.
        """
        log.info(f"Organizing torchvision dataset from temporary location {source_path}...")

        # --- Identify the actual data folder created by torchvision ---
        # It's often inside source_path, named like the dataset or a known variant.
        actual_data_folder = None
        potential_folder_name = KNOWN_TORCHVISION_FOLDERS.get(self.torchvision_dataset_name)

        if potential_folder_name and os.path.isdir(os.path.join(source_path, potential_folder_name)):
            actual_data_folder = os.path.join(source_path, potential_folder_name)
            log.info(f"Found known data folder: {actual_data_folder}")
        else:
            # If not found via known names, look for a single directory inside source_path
            # (excluding potential hidden files/folders)
            items_in_source = [d for d in os.listdir(source_path) if not d.startswith('.')]
            if len(items_in_source) == 1 and os.path.isdir(os.path.join(source_path, items_in_source[0])):
                actual_data_folder = os.path.join(source_path, items_in_source[0])
                log.info(f"Found single data folder: {actual_data_folder}")
            else:
                # Fallback: Assume the source_path itself contains the data files directly
                log.warning(f"Could not identify a unique data subfolder in {source_path}. Assuming data is directly inside.")
                actual_data_folder = source_path # Use the temp root itself

        # --- Move data to final destination ---
        if os.path.exists(self.dataset_final_path):
             log.warning(f"Final dataset path {self.dataset_final_path} already exists. Removing before moving organized data.")
             try:
                  shutil.rmtree(self.dataset_final_path)
             except OSError as e:
                  log.error(f"Failed to remove existing final path {self.dataset_final_path}: {e}")
                  return False # Cannot proceed

        try:
            log.info(f"Moving '{actual_data_folder}' to '{self.dataset_final_path}'")
            shutil.move(actual_data_folder, self.dataset_final_path)
            log.info("Move successful.")
        except Exception as e:
            log.error(f"Failed to move data from {actual_data_folder} to {self.dataset_final_path}: {e}")
            return False

        # --- Standardize internal structure (optional but recommended) ---
        try:
            self._standardize_structure(self.dataset_final_path)
        except Exception as e:
             log.error(f"Error during structure standardization for {self.dataset_name}: {e}", exc_info=True)
             # Don't necessarily fail the whole process if standardization fails, but log it.
             log.warning("Dataset moved, but internal structure standardization failed.")

        return True # Organization (move) was successful, even if standardization had issues


    def _standardize_structure(self, base_path):
        """Attempts to standardize common torchvision dataset structures within base_path."""
        log.info(f"Attempting to standardize structure within {base_path} for {self.torchvision_dataset_name}...")

        # --- VOC Specific ---
        if self.torchvision_dataset_name in ['VOCDetection', 'VOCSegmentation']:
            # Assumes base_path is now the VOCdevkit folder
            voc_devkit_path = base_path
            voc_year_folder = None
            for item in os.listdir(voc_devkit_path):
                if item.startswith("VOC") and os.path.isdir(os.path.join(voc_devkit_path, item)):
                    voc_year_folder = os.path.join(voc_devkit_path, item)
                    break
            if not voc_year_folder:
                log.warning(f"Could not find VOC year folder (e.g., VOC2007) inside {voc_devkit_path}. Skipping standardization.")
                return

            log.info(f"Standardizing VOC structure for {voc_year_folder}")
            # Create standard structure dirs
            train_img_dir = os.path.join(base_path, 'train', 'images')
            train_ann_dir = os.path.join(base_path, 'train', 'annotations')
            val_img_dir = os.path.join(base_path, 'val', 'images')
            val_ann_dir = os.path.join(base_path, 'val', 'annotations')
            test_img_dir = os.path.join(base_path, 'test', 'images')
            test_ann_dir = os.path.join(base_path, 'test', 'annotations') # May remain empty

            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(train_ann_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(val_ann_dir, exist_ok=True)
            os.makedirs(test_img_dir, exist_ok=True)
            os.makedirs(test_ann_dir, exist_ok=True)

            # Source paths within VOC year folder
            imgset_folder = 'Segmentation' if self.task_type == 'segmentation' else 'Main'
            imgset_path = os.path.join(voc_year_folder, 'ImageSets', imgset_folder)
            seg_class_path = os.path.join(voc_year_folder, 'SegmentationClass')
            jpeg_path = os.path.join(voc_year_folder, 'JPEGImages')
            ann_xml_path = os.path.join(voc_year_folder, 'Annotations')

            def process_voc_split(split_name, target_img_dir, target_ann_dir):
                split_file = os.path.join(imgset_path, f"{split_name}.txt")
                # Fallback logic if split file not in expected folder (e.g., Segmentation vs Main)
                if not os.path.exists(split_file) and imgset_folder != 'Main':
                     split_file_main = os.path.join(voc_year_folder, 'ImageSets', 'Main', f"{split_name}.txt")
                     if os.path.exists(split_file_main):
                          log.debug(f"Split file {split_file} not found, using {split_file_main} instead.")
                          split_file = split_file_main
                     else:
                          log.warning(f"Split file not found: {split_file} (and not in Main)")
                          return 0 # Return count 0

                if not os.path.exists(split_file):
                     log.warning(f"Split file not found: {split_file}")
                     return 0

                count = 0
                with open(split_file, 'r') as f:
                    image_ids = [line.strip().split()[0] for line in f if line.strip()]

                for img_id in tqdm(image_ids, desc=f"Standardizing VOC {split_name}", leave=False, unit="file"):
                    # Copy image
                    src_img = os.path.join(jpeg_path, f"{img_id}.jpg")
                    dst_img = os.path.join(target_img_dir, f"{img_id}.jpg")
                    if os.path.exists(src_img):
                         shutil.copy2(src_img, dst_img)
                         count += 1
                    else: log.debug(f"Image source not found: {src_img}")

                    # Copy annotation
                    if self.task_type == 'segmentation':
                        src_ann = os.path.join(seg_class_path, f"{img_id}.png")
                        dst_ann = os.path.join(target_ann_dir, f"{img_id}.png")
                    else: # Detection uses XML
                        src_ann = os.path.join(ann_xml_path, f"{img_id}.xml")
                        dst_ann = os.path.join(target_ann_dir, f"{img_id}.xml")

                    if os.path.exists(src_ann):
                         shutil.copy2(src_ann, dst_ann)
                    else: log.debug(f"Annotation source not found: {src_ann}")
                return count

            # Process splits based on common patterns or kwargs
            processed_count = 0
            processed_count += process_voc_split('train', train_img_dir, train_ann_dir)
            processed_count += process_voc_split('val', val_img_dir, val_ann_dir)
            # Handle trainval logic: often means combining train+val into the train split
            if os.path.exists(os.path.join(imgset_path, 'trainval.txt')):
                 log.info("Processing 'trainval' split (combining into 'train')...")
                 processed_count += process_voc_split('trainval', train_img_dir, train_ann_dir) # Copy trainval to train
                 # Optionally remove separate val split if trainval is the primary training set?
                 # shutil.rmtree(val_img_dir)
                 # shutil.rmtree(val_ann_dir)

            processed_count += process_voc_split('test', test_img_dir, test_ann_dir)

            log.info(f"VOC dataset standardization complete. Processed approx {processed_count} images.")
            # Optionally remove original VOC year folder structure inside base_path
            # log.info(f"Removing original structure in {voc_year_folder}")
            # shutil.rmtree(voc_year_folder)


        # --- OxfordPets / Flowers102 / StanfordDogs ---
        elif self.torchvision_dataset_name in ['OxfordIIITPet', 'Flowers102', 'StanfordDogs']:
             # These often download into 'images' and 'annotations' folders already
             # Goal: Ensure structure is base_path/split/images and base_path/split/annotations|masks
             log.info(f"Standardizing structure for {self.torchvision_dataset_name}...")
             images_src = os.path.join(base_path, 'images')
             annotations_src = os.path.join(base_path, 'annotations') # Pets: trimaps, Flowers: N/A?, Dogs: XMLs?
             lists_src = os.path.join(base_path, 'lists') # Stanford Dogs uses lists for splits

             if not os.path.isdir(images_src):
                  log.warning(f"Could not find 'images' directory in {base_path}. Standardization skipped.")
                  return

             # --- Stanford Dogs Specific Split ---
             if self.torchvision_dataset_name == 'StanfordDogs' and os.path.isdir(lists_src):
                  log.info("Applying Stanford Dogs train/test split based on lists...")
                  train_img_dir = os.path.join(base_path, 'train', 'images')
                  test_img_dir = os.path.join(base_path, 'test', 'images')
                  train_ann_dir = os.path.join(base_path, 'train', 'annotations')
                  test_ann_dir = os.path.join(base_path, 'test', 'annotations')
                  # Annotations are often separate XML files per image, need to find them
                  dog_ann_src = annotations_src if os.path.isdir(annotations_src) else os.path.join(base_path, 'Annotation') # Check actual name

                  os.makedirs(train_img_dir, exist_ok=True)
                  os.makedirs(test_img_dir, exist_ok=True)
                  has_annotations = os.path.isdir(dog_ann_src)
                  if has_annotations:
                       os.makedirs(train_ann_dir, exist_ok=True)
                       os.makedirs(test_ann_dir, exist_ok=True)

                  def process_dog_split(list_file, target_img_dir, target_ann_dir):
                      split_name = os.path.splitext(os.path.basename(list_file))[0].replace('_list','')
                      if not os.path.exists(list_file):
                           log.warning(f"Dog split file not found: {list_file}")
                           return 0
                      count = 0
                      with open(list_file, 'r') as f:
                           image_files = [line.strip() for line in f]
                      for img_rel_path in tqdm(image_files, desc=f"Standardizing Dogs {split_name}", leave=False, unit="file"):
                           src_img = os.path.join(images_src, img_rel_path) # Images are inside 'images' dir
                           dst_img = os.path.join(target_img_dir, img_rel_path)
                           if os.path.exists(src_img):
                                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                                shutil.move(src_img, dst_img) # Move from original images dir
                                count += 1

                           # Move corresponding annotation if exists
                           if has_annotations:
                                # Annotation path often mirrors image path but under Annotation/
                                # e.g. images/n02085620-Chihuahua/n02085620_10074.jpg -> Annotation/n02085620-Chihuahua/n02085620_10074
                                ann_rel_path_base = img_rel_path.split('.')[0] # Remove extension
                                src_ann = os.path.join(dog_ann_src, ann_rel_path_base) # No extension needed for folder name? Check format.
                                dst_ann = os.path.join(target_ann_dir, ann_rel_path_base)

                                if os.path.exists(src_ann): # Check if annotation exists
                                     os.makedirs(os.path.dirname(dst_ann), exist_ok=True)
                                     shutil.move(src_ann, dst_ann) # Move from Annotation dir
                                else:
                                     log.debug(f"Annotation not found for {img_rel_path} at {src_ann}")
                      return count

                  train_count = process_dog_split(os.path.join(lists_src, 'train_list.txt'), train_img_dir, train_ann_dir)
                  test_count = process_dog_split(os.path.join(lists_src, 'test_list.txt'), test_img_dir, test_ann_dir)

                  # Remove original structures if they are now empty
                  log.info(f"Stanford Dogs train/test split applied. Train: {train_count}, Test: {test_count}")
                  try: os.rmdir(images_src)
                  except OSError: log.debug(f"Could not remove original images dir {images_src} (might not be empty).")
                  if has_annotations:
                       try: os.rmdir(dog_ann_src)
                       except OSError: log.debug(f"Could not remove original annotation dir {dog_ann_src} (might not be empty).")
                  shutil.rmtree(lists_src)


             # --- OxfordPets / Flowers102 (Simpler structure) ---
             elif self.torchvision_dataset_name in ['OxfordIIITPet', 'Flowers102']:
                  # Assume single split or pre-split data. Move to 'all_data' structure.
                  log.info(f"Moving {self.torchvision_dataset_name} data to 'all_data' structure.")
                  all_img_dir = os.path.join(base_path, 'all_data', 'images')
                  all_ann_dir = os.path.join(base_path, 'all_data', 'masks' if self.torchvision_dataset_name == 'OxfordIIITPet' else 'annotations') # Pets use masks (trimaps)

                  os.makedirs(all_img_dir, exist_ok=True)
                  for item in os.listdir(images_src): # Move contents
                       shutil.move(os.path.join(images_src, item), os.path.join(all_img_dir, item))
                  os.rmdir(images_src)

                  if os.path.isdir(annotations_src):
                       os.makedirs(all_ann_dir, exist_ok=True)
                       for item in os.listdir(annotations_src): # Move contents
                            shutil.move(os.path.join(annotations_src, item), os.path.join(all_ann_dir, item))
                       os.rmdir(annotations_src)
                  log.info(f"{self.torchvision_dataset_name} moved to 'all_data'.")


        # --- CIFAR / MNIST ---
        elif self.torchvision_dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
             # Data is in specific files (batches.meta, data_batch_*, *.pt).
             # No folder reorganization needed. Visualizer handles loading these files.
             log.info(f"{self.torchvision_dataset_name} uses specific file formats. No folder standardization applied.")
             pass

        # --- Cityscapes ---
        elif self.torchvision_dataset_name == 'Cityscapes':
             # Requires specific handling of gtFine/leftImg8bit folders and splits
             log.info("Standardizing Cityscapes structure...")
             gt_fine_src = os.path.join(base_path, 'gtFine')
             left_img_src = os.path.join(base_path, 'leftImg8bit')

             if not os.path.isdir(gt_fine_src) or not os.path.isdir(left_img_src):
                  log.warning(f"Could not find 'gtFine' or 'leftImg8bit' directories in {base_path}. Skipping Cityscapes standardization.")
                  return

             for split in ['train', 'val', 'test']:
                  split_img_dir = os.path.join(base_path, split, 'images')
                  split_ann_dir = os.path.join(base_path, split, 'annotations') # Masks are typically *_labelIds.png or polygons.json
                  os.makedirs(split_img_dir, exist_ok=True)
                  os.makedirs(split_ann_dir, exist_ok=True)

                  # Source directories for this split
                  src_img_split_dir = os.path.join(left_img_src, split)
                  src_ann_split_dir = os.path.join(gt_fine_src, split)

                  if not os.path.isdir(src_img_split_dir):
                       log.warning(f"Source image directory not found for split '{split}': {src_img_split_dir}")
                       continue

                  # Move images (recursive search within city folders)
                  img_count = 0
                  for city_folder in os.listdir(src_img_split_dir):
                       src_city_img_path = os.path.join(src_img_split_dir, city_folder)
                       if os.path.isdir(src_city_img_path):
                            for fname in os.listdir(src_city_img_path):
                                 if fname.endswith('_leftImg8bit.png'):
                                      shutil.move(os.path.join(src_city_img_path, fname), os.path.join(split_img_dir, fname))
                                      img_count += 1
                  log.info(f"Moved {img_count} images for Cityscapes split '{split}'.")


                  # Move annotations (recursive search within city folders)
                  ann_count = 0
                  if os.path.isdir(src_ann_split_dir):
                       for city_folder in os.listdir(src_ann_split_dir):
                            src_city_ann_path = os.path.join(src_ann_split_dir, city_folder)
                            if os.path.isdir(src_city_ann_path):
                                 for fname in os.listdir(src_city_ann_path):
                                      # Move common annotation types
                                      if fname.endswith(('_gtFine_labelIds.png', '_gtFine_instanceIds.png', '_gtFine_polygons.json', '_gtFine_color.png')):
                                           shutil.move(os.path.join(src_city_ann_path, fname), os.path.join(split_ann_dir, fname))
                                           ann_count += 1
                       log.info(f"Moved {ann_count} annotation files for Cityscapes split '{split}'.")
                  else:
                       log.warning(f"Source annotation directory not found for split '{split}': {src_ann_split_dir}")

             # Remove original gtFine and leftImg8bit folders
             log.info("Removing original Cityscapes directory structure...")
             shutil.rmtree(gt_fine_src)
             shutil.rmtree(left_img_src)
             log.info("Cityscapes standardization complete.")


        else:
             log.info(f"No specific standardization logic implemented for {self.torchvision_dataset_name}. Using structure as downloaded.")