import os
import random
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image # Keep for potential future use (e.g., different mask formats)

# Import from sibling modules
from .utils import find_image_files, find_annotation_file
from .parsers import parse_voc_xml, parse_coco_json, parse_yolo_txt
from .loaders import load_cifar_data, load_mnist_data

log = logging.getLogger(__name__)

class DatasetVisualiser:
    """
    Loads and visualizes image datasets organized by DataRetriever.
    Handles different dataset structures and task types.
    """
    def __init__(self, dataset_path, task_type, dataset_name=None):
        """
        Initializes the visualiser.

        Args:
            dataset_path (str): Path to the root directory of the organized dataset.
            task_type (str): Type of task ('classification', 'object_detection', 'segmentation', etc.).
            dataset_name (str, optional): A hint for dataset type (e.g., 'cifar10', 'mnist')
                                          to trigger specific loaders if structure isn't standard.
                                          Defaults to the base name of dataset_path.
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.task_type = task_type
        self.dataset_name = dataset_name or os.path.basename(self.dataset_path)
        self.dataset_structure = {} # {split: [(img_data/path, annot_data/path/None), ...]}
        self.class_names = None # Optional: list of class names

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found or is not a directory: {self.dataset_path}")

        log.info(f"Initializing visualiser for: {self.dataset_name} ({self.task_type}) at {self.dataset_path}")
        self._load_dataset_structure()

    def _load_dataset_structure(self):
        """Loads the dataset structure based on task type, dataset name hint, and common conventions."""
        log.info("Loading dataset structure...")

        # --- Handle specific dataset loaders first (CIFAR, MNIST) ---
        if self.dataset_name.startswith('cifar'):
            self.dataset_structure, self.class_names = load_cifar_data(self.dataset_path, self.dataset_name)
            if self.dataset_structure: log.info("CIFAR data loaded.")
            return
        if self.dataset_name.startswith('mnist') or self.dataset_name.startswith('fashion_mnist'):
            self.dataset_structure, self.class_names = load_mnist_data(self.dataset_path, self.dataset_name)
            if self.dataset_structure: log.info("MNIST/FashionMNIST data loaded.")
            return

        # --- Handle folder-based datasets (standard structure) ---
        # Expected structure: dataset_path/split/type_specific_folders
        possible_splits = ['train', 'test', 'val', 'all_data']
        found_splits = False

        for split in possible_splits:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.isdir(split_path):
                log.debug(f"Split directory not found: {split_path}")
                continue

            found_splits = True
            self.dataset_structure[split] = []
            log.info(f"Processing split: {split}")

            # Define expected subdirectories based on task type
            images_dir = os.path.join(split_path, 'images')
            annotations_dir = os.path.join(split_path, 'annotations')
            masks_dir = os.path.join(split_path, 'masks') # Common alternative for segmentation

            # --- Classification ---
            if self.task_type == 'classification':
                # Structure 1: split/class_name/image.jpg
                class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d not in ['images', 'annotations', 'masks']]
                if class_folders:
                    log.info(f"Found class folders in {split}: {class_folders}")
                    if self.class_names is None: self.class_names = sorted(class_folders)
                    for class_name in class_folders:
                        class_path = os.path.join(split_path, class_name)
                        image_files = find_image_files(class_path)
                        for img_path in image_files:
                            self.dataset_structure[split].append((img_path, class_name)) # Annot is class name
                # Structure 2: split/images/image.jpg (class info might be in a separate file - not handled here)
                elif os.path.isdir(images_dir):
                     log.warning(f"No class folders found in {split_path}, checking {images_dir}. Class info might be missing.")
                     image_files = find_image_files(images_dir)
                     for img_path in image_files:
                          self.dataset_structure[split].append((img_path, "unknown")) # Assign dummy class
                # Structure 3: split/image.jpg (images directly in split folder)
                else:
                     log.warning(f"No class or images folders found in {split_path}. Checking for images directly in split folder.")
                     image_files = find_image_files(split_path)
                     for img_path in image_files:
                          self.dataset_structure[split].append((img_path, "unknown"))

            # --- Object Detection ---
            elif self.task_type == 'object_detection':
                # Structure: split/images/image.jpg, split/annotations/image.xml|json|txt
                if not os.path.isdir(images_dir):
                    log.warning(f"'images' directory not found in {split_path}. Skipping this split for detection.")
                    continue
                has_annotations = os.path.isdir(annotations_dir)
                if not has_annotations:
                    log.warning(f"'annotations' directory not found in {split_path}. Visualization will only show images.")

                image_files = find_image_files(images_dir)
                annot_exts = ['.xml', '.json', '.txt'] # Common detection formats
                for img_path in image_files:
                    annotation_path = None
                    if has_annotations:
                        annotation_path = find_annotation_file(img_path, annotations_dir, annot_exts)
                        if not annotation_path: log.debug(f"No annotation found for {os.path.basename(img_path)} in {annotations_dir}")
                    self.dataset_structure[split].append((img_path, annotation_path)) # Annot is path or None

            # --- Segmentation ---
            elif self.task_type == 'segmentation':
                # Structure: split/images/image.jpg, split/masks/image.png (or annotations/)
                if not os.path.isdir(images_dir):
                    log.warning(f"'images' directory not found in {split_path}. Skipping this split for segmentation.")
                    continue

                ann_dir_to_use = None
                if os.path.isdir(masks_dir):
                    ann_dir_to_use = masks_dir
                    log.debug(f"Using 'masks' directory for segmentation annotations: {masks_dir}")
                elif os.path.isdir(annotations_dir):
                     log.info(f"Using 'annotations' directory for segmentation annotations: {annotations_dir}")
                     ann_dir_to_use = annotations_dir
                else:
                    log.warning(f"Masks/annotations directory not found in {split_path}. Visualization will only show images.")

                image_files = find_image_files(images_dir)
                # Common mask formats (often images themselves)
                mask_exts = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']
                for img_path in image_files:
                    mask_path = None
                    if ann_dir_to_use:
                        mask_path = find_annotation_file(img_path, ann_dir_to_use, mask_exts)
                        if not mask_path: log.debug(f"No mask found for {os.path.basename(img_path)} in {ann_dir_to_use}")
                    self.dataset_structure[split].append((img_path, mask_path)) # Annot is mask path or None

            # --- Unlabelled / Other ---
            elif self.task_type in ['unlabelled', 'other']:
                 # Structure: split/images/ or just files in split/
                 img_dir_to_check = images_dir if os.path.isdir(images_dir) else split_path
                 image_files = find_image_files(img_dir_to_check)
                 for img_path in image_files:
                     self.dataset_structure[split].append((img_path, None)) # No annotations expected

        # --- Fallback: Check root dataset folder directly if no standard splits found ---
        if not found_splits:
             log.warning(f"No standard splits (train/test/val/all_data) found in {self.dataset_path}. Checking root directory structure.")
             self.dataset_structure['all_data'] = []
             # Reuse logic from above, but apply to root path instead of split_path
             images_dir = os.path.join(self.dataset_path, 'images')
             annotations_dir = os.path.join(self.dataset_path, 'annotations')
             masks_dir = os.path.join(self.dataset_path, 'masks')

             if self.task_type == 'classification':
                  class_folders = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d)) and d not in ['images', 'annotations', 'masks']]
                  if class_folders:
                       if self.class_names is None: self.class_names = sorted(class_folders)
                       for class_name in class_folders:
                            class_path = os.path.join(self.dataset_path, class_name)
                            image_files = find_image_files(class_path)
                            for img_path in image_files:
                                 self.dataset_structure['all_data'].append((img_path, class_name))
                  else: # Images directly in root or root/images
                       img_dir_to_check = images_dir if os.path.isdir(images_dir) else self.dataset_path
                       image_files = find_image_files(img_dir_to_check)
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
                  img_dir_to_check = images_dir if os.path.isdir(images_dir) else self.dataset_path
                  image_files = find_image_files(img_dir_to_check)
                  for img_path in image_files:
                       self.dataset_structure['all_data'].append((img_path, None))


        if not self.dataset_structure or all(not items for items in self.dataset_structure.values()):
            log.error(f"Could not find any image data in the expected structure within {self.dataset_path} for task type {self.task_type}.")
        else:
             total_items = sum(len(items) for items in self.dataset_structure.values())
             log.info(f"Dataset structure loaded successfully. Found {total_items} items across {len(self.dataset_structure)} splits.")


    def run_eda(self, show_plots=False):
        """Performs basic Exploratory Data Analysis."""
        log.info(f"--- Basic EDA for {self.dataset_name} ---")
        total_items = 0
        split_counts = {}
        class_counts = {} # For classification

        if not self.dataset_structure:
             log.error("Dataset structure not loaded. Cannot run EDA.")
             return

        for split, items in self.dataset_structure.items():
            count = len(items)
            split_counts[split] = count
            total_items += count
            log.info(f"Split '{split}': {count} items found.")

            if self.task_type == 'classification':
                for _, annot_data in items:
                    # Annot data is class name for classification
                    class_name = annot_data if isinstance(annot_data, str) else "unknown"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        log.info(f"Total items found across all splits: {total_items}")

        if self.task_type == 'classification' and class_counts:
            log.info("Class distribution:")
            # Use self.class_names if available for order, otherwise sort keys
            sorted_classes = self.class_names if self.class_names else sorted(class_counts.keys())
            for class_name in sorted_classes:
                 if class_name in class_counts: # Only show classes present in the loaded data
                      log.info(f"  - {class_name}: {class_counts[class_name]}")

            # Plot class distribution
            if show_plots:
                try:
                    plt.figure(figsize=(max(10, len(sorted_classes)*0.5), 5)) # Adjust width
                    plot_labels = [cn for cn in sorted_classes if cn in class_counts]
                    plot_values = [class_counts[cn] for cn in plot_labels]
                    plt.bar(plot_labels, plot_values)
                    plt.title(f'Class Distribution - {self.dataset_name}')
                    plt.xlabel('Class')
                    plt.ylabel('Number of Items')
                    plt.xticks(rotation=45, ha='right', fontsize=8)
                    plt.tight_layout()
                    # Don't call plt.show() here, let the caller handle it or view_samples handle it
                    # plt.show()
                except Exception as e:
                     log.error(f"Failed to generate class distribution plot: {e}")
            else:
                 log.info("Class distribution plot generation skipped (show_plots=False).")


        # TODO: Add more EDA: image size distribution, annotation counts per image (obj det), etc.
        log.info("-----------------")


    def view_samples(self, num_samples=9, cols=3, splits=None, show_plots=True):
        """
        Displays a grid of sample images with annotations if applicable.

        Args:
            num_samples (int): Number of samples to display.
            cols (int): Number of columns in the display grid.
            splits (list, optional): Specific splits to sample from (e.g., ['train', 'test']).
                                     Defaults to all available splits.
            show_plots (bool): If True, calls plt.show() at the end to display the plot.
                               If False, the plot is generated but not shown (useful for testing).
        """
        log.info(f"Preparing to view {num_samples} sample images...")

        # Flatten the list of items from specified splits or all splits
        items_to_sample = []
        target_splits = splits or self.dataset_structure.keys()
        for split in target_splits:
            if split in self.dataset_structure:
                items_to_sample.extend(self.dataset_structure[split])
            else:
                 log.warning(f"Requested split '{split}' not found in loaded data.")


        if not items_to_sample:
            log.error("No items found in the specified splits to visualize.")
            return

        # Ensure num_samples is not greater than the total number of items
        num_samples = min(num_samples, len(items_to_sample))
        if num_samples <= 0:
            log.warning("Number of samples must be positive.")
            return

        samples = random.sample(items_to_sample, num_samples)

        rows = (num_samples + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 5, rows * 5)) # Adjust figure size as needed

        for i, (img_data, annot_data) in enumerate(samples):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.axis('off')
            title = "Sample"
            img_rgb = None
            img_height, img_width = None, None

            try:
                # --- Load Image ---
                if isinstance(img_data, str) and os.path.exists(img_data): # Image is a path
                    img_path = img_data
                    title = os.path.basename(img_path)
                    # Use OpenCV, handles more formats robustly than PIL sometimes
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        log.error(f"Failed to load image: {img_path}")
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
                         log.error(f"Unsupported numpy array shape for image: {img_array.shape}")
                         ax.set_title(f"Unsupported array shape\n{title}", fontsize=8)
                         continue
                    img_height, img_width = img_rgb.shape[:2]
                else:
                    log.error(f"Invalid image data type: {type(img_data)}")
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
                        img_filename = os.path.basename(img_data) if isinstance(img_data, str) else None
                        if img_filename:
                             boxes, _ = parse_coco_json(annotation_path, img_filename)
                        else: log.warning("Cannot parse COCO JSON for numpy image data without filename context.")
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
                         # Load mask using OpenCV (consistent with image loading)
                         mask = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
                         if mask is None:
                             log.error(f"Failed to load mask: {annotation_path}")
                         else:
                             # Resize mask to match image if necessary (common issue)
                             if mask.shape[0] != img_height or mask.shape[1] != img_width:
                                  log.warning(f"Mask size {mask.shape[:2]} differs from image size {(img_height, img_width)}. Resizing mask.")
                                  mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST) # Use nearest neighbor for masks

                             # Simple overlay logic
                             if len(mask.shape) == 3: # Color mask (assume BGR format from cv2.imread)
                                 mask_rgba = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
                                 mask_rgba[:, :, 3] = 128 # Set alpha
                                 ax.imshow(mask_rgba, alpha=0.5) # Overlay with transparency
                             elif len(mask.shape) == 2: # Grayscale mask (class IDs)
                                 # Create a colored overlay based on class IDs
                                 num_classes = np.max(mask) + 1
                                 # Use a qualitative colormap suitable for segmentation
                                 # Avoid relying solely on viridis which implies order
                                 colors = plt.cm.get_cmap('tab20', num_classes) if num_classes > 10 else plt.cm.get_cmap('tab10', num_classes)
                                 mask_colored = np.zeros((*mask.shape, 4)) # RGBA

                                 # Apply colors, skipping background (class 0 often)
                                 for class_id in range(1, num_classes): # Skip 0
                                     mask_colored[mask == class_id] = colors(class_id / max(1, num_classes - 1)) # Normalize ID for colormap index

                                 mask_colored[:, :, 3] = (mask > 0) * 0.5 # Set alpha only for non-background pixels
                                 ax.imshow(mask_colored) # Overlay colored mask

                     except Exception as e:
                         log.error(f"Error processing mask {annotation_path}: {e}")

                ax.set_title(title, fontsize=8)

            except Exception as e:
                log.error(f"Error processing or plotting sample: {e}", exc_info=True)
                ax.set_title(f"Error processing sample", fontsize=8)

        fig.tight_layout()
        if show_plots:
            log.debug("Displaying plot window...")
            plt.show()
        else:
             log.debug("Closing plot figure (show_plots=False).")
             plt.close(fig)