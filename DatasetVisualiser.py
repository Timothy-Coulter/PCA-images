import os
import argparse
import random
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2  # OpenCV for image loading and drawing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image # For mask handling, potentially loading other formats
import logging
from glob import glob
import pickle
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Annotation Parsers ---

def parse_voc_xml(annotation_path):
    """Parses a PASCAL VOC XML annotation file."""
    boxes = []
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            # Handle potential float values in XML, convert safely
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append({'label': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file {annotation_path}: {e}")
    except (ValueError, TypeError) as e:
         logging.error(f"Error converting coordinate to int in {annotation_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error parsing VOC XML {annotation_path}: {e}")
    return boxes

def parse_coco_json(annotation_path, image_filename):
    """Parses COCO JSON format (expects a single JSON for the whole split)."""
    # This requires loading the entire JSON and finding annotations linked to the image_filename.
    # This is complex and usually done with helper libraries (pycocotools).
    # Placeholder implementation.
    boxes = []
    segments = [] # For segmentation
    logging.warning(f"COCO JSON parsing ({annotation_path}) is a placeholder. Requires pycocotools or similar for proper implementation.")
    try:
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        # Find image ID
        image_id = None
        for img_info in coco_data.get('images', []):
            if img_info['file_name'] == image_filename:
                image_id = img_info['id']
                break

        if image_id is None:
            logging.warning(f"Image filename '{image_filename}' not found in COCO JSON {annotation_path}")
            return boxes, segments # Return empty lists

        # Find annotations for this image ID
        for ann_info in coco_data.get('annotations', []):
            if ann_info['image_id'] == image_id:
                category_id = ann_info['category_id']
                # Find category name (optional, requires 'categories' list in JSON)
                label = f"cat_{category_id}"
                for cat_info in coco_data.get('categories', []):
                     if cat_info['id'] == category_id:
                          label = cat_info['name']
                          break

                # Bounding box [xmin, ymin, width, height]
                if 'bbox' in ann_info:
                    bbox = ann_info['bbox']
                    xmin, ymin, w, h = map(int, bbox)
                    boxes.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmin + w, 'ymax': ymin + h})

                # Segmentation (list of polygons or RLE)
                if 'segmentation' in ann_info:
                     # Handling segmentation is complex (polygons or RLE)
                     # Add placeholder logic or skip for now
                     segments.append({'label': label, 'data': ann_info['segmentation']})


    except json.JSONDecodeError as e:
        logging.error(f"Error decoding COCO JSON file {annotation_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error parsing COCO JSON {annotation_path}: {e}")

    # For visualization, we primarily need boxes now. Segmentation drawing is complex.
    return boxes, segments


def parse_yolo_txt(annotation_path, img_width, img_height):
    """Parses YOLO TXT format (class x_center y_center width height - normalized)."""
    boxes = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5: # Allow for extra columns sometimes present
                    class_id = int(parts[0]) # Assuming class ID, need mapping for label name
                    x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:5])

                    # Convert normalized coordinates to absolute pixel values
                    width = width_norm * img_width
                    height = height_norm * img_height
                    xmin = int((x_center_norm * img_width) - (width / 2))
                    ymin = int((y_center_norm * img_height) - (height / 2))
                    xmax = int(xmin + width)
                    ymax = int(ymin + height)

                    # Clamp coordinates to image boundaries
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width - 1, xmax)
                    ymax = min(img_height - 1, ymax)


                    boxes.append({'label': f'class_{class_id}', 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                else:
                    logging.warning(f"Skipping invalid line in YOLO file {annotation_path}: {line.strip()}")
    except ValueError as e:
         logging.error(f"Error converting value in YOLO file {annotation_path}: {e}")
    except Exception as e:
        logging.error(f"Error parsing YOLO TXT file {annotation_path}: {e}")
    return boxes


# --- Data Loading Helpers ---

def find_image_files(directory):
    """Finds common image files recursively in a directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
    return image_files

def find_annotation_file(img_path, annotation_dir, expected_exts):
    """Tries to find a matching annotation file for an image."""
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    for annot_ext in expected_exts:
        potential_annot_path = os.path.join(annotation_dir, img_basename + annot_ext)
        if os.path.exists(potential_annot_path):
            return potential_annot_path
    # Try searching in subdirs of annotation_dir (e.g., VOC annotations)
    for root, _, files in os.walk(annotation_dir):
         for annot_ext in expected_exts:
              potential_annot_path = os.path.join(root, img_basename + annot_ext)
              if os.path.exists(potential_annot_path):
                   return potential_annot_path
    return None

# --- Dataset Specific Loaders ---

def load_cifar_batch(filepath):
    """Loads a CIFAR batch file (pickle format)."""
    try:
        with open(filepath, 'rb') as f:
            # Encoding latin1 is important for Python 3 compatibility with Python 2 pickles
            entry = pickle.load(f, encoding='latin1')
            # Data is NCHW format (N x 3 x 32 x 32), needs transpose to NHWC (N x 32 x 32 x 3) for display
            images = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = entry['labels']
            filenames = entry.get('filenames', [f"img_{i}" for i in range(len(labels))]) # Use index if no filename
            return images, labels, filenames
    except FileNotFoundError:
        logging.error(f"CIFAR batch file not found: {filepath}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading CIFAR batch {filepath}: {e}")
        return None, None, None

def load_mnist_images(filepath):
    """Loads MNIST image file (IDX format)."""
    try:
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and dimensions
            magic, num_images = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
            if magic != 2051: raise ValueError("Invalid MNIST image file magic number")
            num_rows, num_cols = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, num_rows, num_cols)
            return images
    except FileNotFoundError:
        logging.error(f"MNIST image file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading MNIST images {filepath}: {e}")
        return None

def load_mnist_labels(filepath):
    """Loads MNIST label file (IDX format)."""
    try:
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and number of items
            magic, num_items = np.frombuffer(f.read(8), dtype=np.dtype('>i4'))
            if magic != 2049: raise ValueError("Invalid MNIST label file magic number")
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    except FileNotFoundError:
        logging.error(f"MNIST label file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading MNIST labels {filepath}: {e}")
        return None


# --- Main Visualizer Class ---

class DatasetVisualiser:
    def __init__(self, dataset_path, task_type, dataset_name=None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.task_type = task_type
        self.dataset_name = dataset_name or os.path.basename(self.dataset_path)
        self.dataset_structure = {} # {split: [(img_data, annot_data), ...]}
        self.class_names = None # Optional: list of class names for classification/detection

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found or is not a directory: {self.dataset_path}")

        logging.info(f"Initializing visualiser for: {self.dataset_name} ({self.task_type}) at {self.dataset_path}")
        self._load_dataset_structure()

    def _load_dataset_structure(self):
        """Loads the dataset structure based on task type and common conventions."""
        logging.info("Loading dataset structure...")

        # --- Handle file-based datasets first (CIFAR, MNIST) ---
        if self.dataset_name.startswith('cifar'):
            self._load_cifar()
            return
        if self.dataset_name.startswith('mnist') or self.dataset_name.startswith('fashion_mnist'):
            self._load_mnist()
            return

        # --- Handle folder-based datasets ---
        possible_splits = ['train', 'test', 'val', 'all_data'] # Expected subdirs from DataRetriever
        found_splits = False

        for split in possible_splits:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.isdir(split_path):
                continue

            found_splits = True
            self.dataset_structure[split] = []
            logging.info(f"Processing split: {split}")

            images_dir = os.path.join(split_path, 'images')
            annotations_dir = os.path.join(split_path, 'annotations')
            masks_dir = os.path.join(split_path, 'masks') # Common alternative for segmentation

            if self.task_type == 'classification':
                # Structure: split/class_name/image.jpg OR split/images/ (less common)
                class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d != 'images' and d != 'annotations' and d != 'masks']
                if class_folders:
                    logging.info(f"Found class folders: {class_folders}")
                    if self.class_names is None: self.class_names = sorted(class_folders)
                    for class_name in class_folders:
                        class_path = os.path.join(split_path, class_name)
                        image_files = find_image_files(class_path)
                        for img_path in image_files:
                            self.dataset_structure[split].append((img_path, class_name)) # Annot is class name
                elif os.path.isdir(images_dir): # Check for split/images structure
                     logging.warning(f"No class folders found in {split_path}, checking {images_dir}. Class info might be missing.")
                     image_files = find_image_files(images_dir)
                     for img_path in image_files:
                          self.dataset_structure[split].append((img_path, "unknown"))
                else: # Check for images directly in split folder
                     logging.warning(f"No class or images folders found in {split_path}. Checking for images directly.")
                     image_files = find_image_files(split_path)
                     for img_path in image_files:
                          self.dataset_structure[split].append((img_path, "unknown"))


            elif self.task_type == 'object_detection':
                # Structure: split/images/image.jpg, split/annotations/image.xml|json|txt
                if not os.path.isdir(images_dir):
                    logging.warning(f"'images' directory not found in {split_path}. Skipping this split for detection.")
                    continue
                if not os.path.isdir(annotations_dir):
                    logging.warning(f"'annotations' directory not found in {split_path}. Visualization will only show images.")

                image_files = find_image_files(images_dir)
                annot_exts = ['.xml', '.json', '.txt'] # Common detection formats
                for img_path in image_files:
                    annotation_path = None
                    if os.path.isdir(annotations_dir):
                        annotation_path = find_annotation_file(img_path, annotations_dir, annot_exts)
                    self.dataset_structure[split].append((img_path, annotation_path)) # Annot is path or None

            elif self.task_type == 'segmentation':
                # Structure: split/images/image.jpg, split/masks/image.png (or annotations/)
                if not os.path.isdir(images_dir):
                    logging.warning(f"'images' directory not found in {split_path}. Skipping this split for segmentation.")
                    continue

                ann_dir_to_use = None
                if os.path.isdir(masks_dir):
                    ann_dir_to_use = masks_dir
                elif os.path.isdir(annotations_dir):
                     logging.info(f"Found 'annotations' instead of 'masks' in {split_path} for segmentation.")
                     ann_dir_to_use = annotations_dir
                else:
                    logging.warning(f"Masks/annotations directory not found in {split_path}. Visualization will only show images.")

                image_files = find_image_files(images_dir)
                # Common mask formats (often images themselves)
                mask_exts = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']
                for img_path in image_files:
                    mask_path = None
                    if ann_dir_to_use:
                        mask_path = find_annotation_file(img_path, ann_dir_to_use, mask_exts)
                    self.dataset_structure[split].append((img_path, mask_path)) # Annot is mask path or None

            elif self.task_type in ['unlabelled', 'other']:
                 # Structure: split/images/ or just files in split/
                 img_dir_to_check = images_dir if os.path.isdir(images_dir) else split_path
                 image_files = find_image_files(img_dir_to_check)
                 for img_path in image_files:
                     self.dataset_structure[split].append((img_path, None)) # No annotations expected

        # Fallback: If no standard splits found, check root dataset folder directly
        if not found_splits:
             logging.warning(f"No standard splits (train/test/val/all_data) found in {self.dataset_path}. Checking root directory.")
             self.dataset_structure['all_data'] = []
             images_dir = os.path.join(self.dataset_path, 'images')
             annotations_dir = os.path.join(self.dataset_path, 'annotations')
             masks_dir = os.path.join(self.dataset_path, 'masks')

             if self.task_type == 'classification':
                  class_folders = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
                  if class_folders:
                       if self.class_names is None: self.class_names = sorted(class_folders)
                       for class_name in class_folders:
                            class_path = os.path.join(self.dataset_path, class_name)
                            image_files = find_image_files(class_path)
                            for img_path in image_files:
                                 self.dataset_structure['all_data'].append((img_path, class_name))
                  else: # Images directly in root
                       image_files = find_image_files(self.dataset_path)
                       for img_path in image_files:
                            self.dataset_structure['all_data'].append((img_path, "unknown"))

             elif self.task_type == 'object_detection':
                  img_dir_to_check = images_dir if os.path.isdir(images_dir) else self.dataset_path
                  ann_dir_to_check = annotations_dir if os.path.isdir(annotations_dir) else self.dataset_path
                  image_files = find_image_files(img_dir_to_check)
                  annot_exts = ['.xml', '.json', '.txt']
                  for img_path in image_files:
                       annotation_path = find_annotation_file(img_path, ann_dir_to_check, annot_exts)
                       self.dataset_structure['all_data'].append((img_path, annotation_path))

             elif self.task_type == 'segmentation':
                  img_dir_to_check = images_dir if os.path.isdir(images_dir) else self.dataset_path
                  ann_dir_to_check = masks_dir if os.path.isdir(masks_dir) else (annotations_dir if os.path.isdir(annotations_dir) else self.dataset_path)
                  image_files = find_image_files(img_dir_to_check)
                  mask_exts = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']
                  for img_path in image_files:
                       mask_path = find_annotation_file(img_path, ann_dir_to_check, mask_exts)
                       self.dataset_structure['all_data'].append((img_path, mask_path))

             else: # Unlabelled / Other
                  image_files = find_image_files(self.dataset_path)
                  for img_path in image_files:
                       self.dataset_structure['all_data'].append((img_path, None))


        if not self.dataset_structure:
            logging.error(f"Could not find any image data in the expected structure within {self.dataset_path} for task type {self.task_type}.")
        else:
             logging.info("Dataset structure loaded successfully.")


    def _load_cifar(self):
        """Loads CIFAR10/100 data from batch files."""
        logging.info(f"Loading {self.dataset_name} data...")
        is_cifar100 = '100' in self.dataset_name
        label_key = 'fine_labels' if is_cifar100 else 'labels' # CIFAR-100 has fine and coarse

        # Find batch files (common naming conventions)
        batch_files = glob(os.path.join(self.dataset_path, '**', 'data_batch*'), recursive=True)
        test_batch_files = glob(os.path.join(self.dataset_path, '**', 'test_batch*'), recursive=True)

        if not batch_files and not test_batch_files:
             # Check for root folder name used by torchvision download
             tv_folder = 'cifar-100-python' if is_cifar100 else 'cifar-10-batches-py'
             cifar_root = os.path.join(self.dataset_path, tv_folder)
             if os.path.isdir(cifar_root):
                  batch_files = glob(os.path.join(cifar_root, 'data_batch*'))
                  test_batch_files = glob(os.path.join(cifar_root, 'test_batch*'))
             else:
                  logging.error(f"Could not find CIFAR batch files in {self.dataset_path} or expected subfolder {tv_folder}.")
                  return

        # Load meta file for class names
        meta_file = os.path.join(os.path.dirname(batch_files[0] if batch_files else test_batch_files[0]), 'batches.meta' if not is_cifar100 else 'meta')
        if os.path.exists(meta_file):
             with open(meta_file, 'rb') as f:
                  meta_data = pickle.load(f, encoding='latin1')
                  self.class_names = meta_data['fine_label_names' if is_cifar100 else 'label_names']
                  logging.info(f"Loaded {len(self.class_names)} class names for {self.dataset_name}")


        self.dataset_structure['train'] = []
        for batch_file in batch_files:
            images, labels, filenames = load_cifar_batch(batch_file)
            if images is not None:
                for img, label_idx, fname in zip(images, labels, filenames):
                    label = self.class_names[label_idx] if self.class_names and 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
                    # Store image data directly, not path
                    self.dataset_structure['train'].append((img, label))

        self.dataset_structure['test'] = []
        for batch_file in test_batch_files:
            images, labels, filenames = load_cifar_batch(batch_file)
            if images is not None:
                for img, label_idx, fname in zip(images, labels, filenames):
                    label = self.class_names[label_idx] if self.class_names and 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
                    self.dataset_structure['test'].append((img, label))

        logging.info(f"Loaded {len(self.dataset_structure.get('train',[]))} train and {len(self.dataset_structure.get('test',[]))} test images for {self.dataset_name}.")


    def _load_mnist(self):
        """Loads MNIST/FashionMNIST data from IDX files."""
        logging.info(f"Loading {self.dataset_name} data...")
        # Expected file names (can vary slightly)
        train_img_file = 'train-images-idx3-ubyte.gz'
        train_lbl_file = 'train-labels-idx1-ubyte.gz'
        test_img_file = 't10k-images-idx3-ubyte.gz'
        test_lbl_file = 't10k-labels-idx1-ubyte.gz'

        # Find files, potentially inside subfolders like 'MNIST/raw' or 'FashionMNIST/raw'
        base_paths_to_check = [self.dataset_path,
                               os.path.join(self.dataset_path, 'MNIST', 'raw'),
                               os.path.join(self.dataset_path, 'FashionMNIST', 'raw')]
        found_path = None
        for p in base_paths_to_check:
             if os.path.exists(os.path.join(p, train_img_file)):
                  found_path = p
                  break

        if not found_path:
             logging.error(f"Could not find MNIST/FashionMNIST IDX files in expected locations within {self.dataset_path}")
             return

        train_images = load_mnist_images(os.path.join(found_path, train_img_file))
        train_labels = load_mnist_labels(os.path.join(found_path, train_lbl_file))
        test_images = load_mnist_images(os.path.join(found_path, test_img_file))
        test_labels = load_mnist_labels(os.path.join(found_path, test_lbl_file))

        # Define class names (MNIST: 0-9, FashionMNIST: lookup needed)
        if self.dataset_name.startswith('fashion'):
             self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        else: # MNIST
             self.class_names = [str(i) for i in range(10)]


        self.dataset_structure['train'] = []
        if train_images is not None and train_labels is not None and len(train_images) == len(train_labels):
            for img, label_idx in zip(train_images, train_labels):
                label = self.class_names[label_idx] if self.class_names and 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
                self.dataset_structure['train'].append((img, label)) # Store image data directly

        self.dataset_structure['test'] = []
        if test_images is not None and test_labels is not None and len(test_images) == len(test_labels):
            for img, label_idx in zip(test_images, test_labels):
                label = self.class_names[label_idx] if self.class_names and 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
                self.dataset_structure['test'].append((img, label)) # Store image data directly

        logging.info(f"Loaded {len(self.dataset_structure.get('train',[]))} train and {len(self.dataset_structure.get('test',[]))} test images for {self.dataset_name}.")


    def run_eda(self):
        """Performs basic Exploratory Data Analysis."""
        logging.info(f"--- Basic EDA for {self.dataset_name} ---")
        total_items = 0
        split_counts = {}
        class_counts = {} # For classification

        if not self.dataset_structure:
             logging.error("Dataset structure not loaded. Cannot run EDA.")
             return

        for split, items in self.dataset_structure.items():
            count = len(items)
            split_counts[split] = count
            total_items += count
            logging.info(f"Split '{split}': {count} items found.")

            if self.task_type == 'classification':
                for _, annot_data in items:
                    # Annot data is class name for classification
                    class_name = annot_data if isinstance(annot_data, str) else "unknown"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        logging.info(f"Total items found across all splits: {total_items}")

        if self.task_type == 'classification' and class_counts:
            logging.info("Class distribution:")
            # Use self.class_names if available for order, otherwise sort keys
            sorted_classes = self.class_names if self.class_names else sorted(class_counts.keys())
            for class_name in sorted_classes:
                 if class_name in class_counts: # Only show classes present in the loaded data
                      logging.info(f"  - {class_name}: {class_counts[class_name]}")

            # Plot class distribution
            try:
                plt.figure(figsize=(max(10, len(sorted_classes)*0.5), 5)) # Adjust width based on num classes
                # Ensure keys exist in counts before plotting
                plot_labels = [cn for cn in sorted_classes if cn in class_counts]
                plot_values = [class_counts[cn] for cn in plot_labels]
                plt.bar(plot_labels, plot_values)
                plt.title(f'Class Distribution - {self.dataset_name}')
                plt.xlabel('Class')
                plt.ylabel('Number of Items')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                 logging.error(f"Failed to plot class distribution: {e}")


        # TODO: Add more EDA: image size distribution, annotation counts per image (obj det), etc.
        logging.info("-----------------")


    def view_samples(self, num_samples=9, cols=3, splits=None):
        """Displays a grid of sample images with annotations if applicable."""
        logging.info(f"Preparing to view {num_samples} sample images...")

        # Flatten the list of items from specified splits or all splits
        items_to_sample = []
        target_splits = splits or self.dataset_structure.keys()
        for split in target_splits:
            if split in self.dataset_structure:
                items_to_sample.extend(self.dataset_structure[split])
            else:
                 logging.warning(f"Requested split '{split}' not found in loaded data.")


        if not items_to_sample:
            logging.error("No items found in the specified splits to visualize.")
            return

        # Ensure num_samples is not greater than the total number of items
        num_samples = min(num_samples, len(items_to_sample))
        if num_samples <= 0:
            logging.warning("Number of samples must be positive.")
            return

        samples = random.sample(items_to_sample, num_samples)

        rows = (num_samples + cols - 1) // cols
        plt.figure(figsize=(cols * 5, rows * 5)) # Adjust figure size as needed

        for i, (img_data, annot_data) in enumerate(samples):
            plt.subplot(rows, cols, i + 1)
            ax = plt.gca()
            ax.axis('off')
            title = "Sample"
            img_rgb = None
            img_height, img_width = None, None

            try:
                # --- Load Image ---
                if isinstance(img_data, str) and os.path.exists(img_data): # Image is a path
                    img_path = img_data
                    title = os.path.basename(img_path)
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        logging.error(f"Failed to load image: {img_path}")
                        ax.set_title(f"Error loading:\n{title}", fontsize=8)
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_height, img_width = img_rgb.shape[:2]
                elif isinstance(img_data, np.ndarray): # Image is numpy array (e.g., CIFAR, MNIST)
                    img_array = img_data
                    title = f"{self.dataset_name} sample"
                    if len(img_array.shape) == 2: # Grayscale (MNIST)
                        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3: # RGB (CIFAR)
                         img_rgb = img_array
                    else:
                         logging.error(f"Unsupported numpy array shape for image: {img_array.shape}")
                         ax.set_title(f"Unsupported array shape\n{title}", fontsize=8)
                         continue
                    img_height, img_width = img_rgb.shape[:2]
                else:
                    logging.error(f"Invalid image data type: {type(img_data)}")
                    ax.set_title(f"Invalid image data\n{title}", fontsize=8)
                    continue

                ax.imshow(img_rgb)

                # --- Process Annotations ---
                annotation_path = None
                if isinstance(annot_data, str):
                     if self.task_type == 'classification':
                          title = f"Class: {annot_data}\n{title}"
                     elif os.path.exists(annot_data): # Annotation is a path
                          annotation_path = annot_data
                          title = f"{title}\n+ {os.path.basename(annotation_path)}"
                     # else: annot_data might be class name even for non-classification if structure was odd

                # --- Draw Annotations ---
                if self.task_type == 'object_detection' and annotation_path:
                    boxes = []
                    ext = os.path.splitext(annotation_path)[1].lower()
                    if ext == '.xml':
                        boxes = parse_voc_xml(annotation_path)
                    elif ext == '.json':
                        # Pass image filename if img_data was path, else None
                        img_filename = os.path.basename(img_data) if isinstance(img_data, str) else None
                        if img_filename:
                             boxes, _ = parse_coco_json(annotation_path, img_filename)
                        else:
                             logging.warning("Cannot parse COCO JSON for numpy image data without filename context.")
                    elif ext == '.txt':
                        boxes = parse_yolo_txt(annotation_path, img_width, img_height)

                    # Draw boxes
                    for box in boxes:
                        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                        label = box.get('label', 'N/A')
                        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(xmin, ymin - 5, label, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0))

                elif self.task_type == 'segmentation' and annotation_path:
                     try:
                         mask = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
                         if mask is None:
                             logging.error(f"Failed to load mask: {annotation_path}")
                         else:
                             # Simple overlay logic (can be improved)
                             if len(mask.shape) == 3: # Color mask
                                 mask_rgba = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
                                 mask_rgba[:, :, 3] = 128 # Set alpha
                                 ax.imshow(mask_rgba, alpha=0.5)
                             elif len(mask.shape) == 2: # Grayscale mask (class IDs)
                                 num_classes = np.max(mask) + 1
                                 colors = plt.cm.get_cmap('viridis', num_classes)
                                 mask_colored = np.zeros((*mask.shape, 4)) # RGBA
                                 for class_id in range(num_classes):
                                     if class_id == 0: continue # Skip background
                                     mask_colored[mask == class_id] = colors(class_id / max(1, (num_classes -1))) # Normalize ID

                                 mask_colored[:, :, 3] = (mask > 0) * 0.5 # Alpha for non-bg
                                 ax.imshow(mask_colored)

                     except Exception as e:
                         logging.error(f"Error processing mask {annotation_path}: {e}")

                ax.set_title(title, fontsize=8)

            except Exception as e:
                logging.error(f"Error processing or plotting sample: {e}", exc_info=True) # Add traceback
                ax.set_title(f"Error processing sample", fontsize=8)

        plt.tight_layout()
        plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image datasets loaded by DataRetriever.")

    parser.add_argument("dataset_path", type=str, help="Path to the root directory of the dataset.")
    parser.add_argument("-t", "--task-type", type=str, required=True,
                        choices=['classification', 'object_detection', 'segmentation', 'unlabelled', 'other'],
                        help="Type of computer vision task the dataset is for.")
    parser.add_argument("--dataset-name", type=str, help="Optional name for the dataset (used for specific loaders like CIFAR/MNIST if path doesn't make it obvious).")
    parser.add_argument("--view-samples", action=argparse.BooleanOptionalAction, default=True, help="View sample images (default: True). Use --no-view-samples to disable.")
    parser.add_argument("--run-eda", action=argparse.BooleanOptionalAction, default=True, help="Run basic EDA (default: True). Use --no-run-eda to disable.")
    parser.add_argument("-n", "--num-samples", type=int, default=9, help="Number of samples to visualize (default: 9).")
    parser.add_argument("--splits", nargs='+', default=None, help="Specific splits to visualize (e.g., 'train' 'test'). Default is all available splits.")


    args = parser.parse_args()

    try:
        visualiser = DatasetVisualiser(args.dataset_path, args.task_type, args.dataset_name)

        if args.run_eda:
            visualiser.run_eda()

        if args.view_samples:
            visualiser.view_samples(num_samples=args.num_samples, splits=args.splits)

    except FileNotFoundError as e:
        logging.error(e)
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        exit(1)