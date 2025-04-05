import os
import logging
import pickle
import gzip
import numpy as np
from glob import glob

log = logging.getLogger(__name__)

# --- CIFAR Loader ---

def load_cifar_batch(filepath):
    """Loads a single CIFAR batch file (pickle format)."""
    try:
        with open(filepath, 'rb') as f:
            # Encoding latin1 is important for Python 3 compatibility with Python 2 pickles
            entry = pickle.load(f, encoding='latin1')
            # Data is NCHW format (N x 3 x 32 x 32), needs transpose to NHWC (N x 32 x 32 x 3) for display
            images = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = entry['labels'] # Use 'labels' for CIFAR-10, handle fine/coarse later if needed
            filenames = entry.get('filenames', [f"img_{i}" for i in range(len(labels))]) # Use index if no filename
            log.debug(f"Loaded {len(images)} images from CIFAR batch: {filepath}")
            return images, labels, filenames
    except FileNotFoundError:
        log.error(f"CIFAR batch file not found: {filepath}")
        return None, None, None
    except Exception as e:
        log.error(f"Error loading CIFAR batch {filepath}: {e}")
        return None, None, None

def load_cifar_data(dataset_path, dataset_name):
    """Loads all CIFAR10/100 data from batch files found in dataset_path."""
    log.info(f"Loading {dataset_name} data from path: {dataset_path}...")
    is_cifar100 = '100' in dataset_name
    label_key = 'fine_labels' if is_cifar100 else 'labels' # Key in meta file

    # Find batch files (common naming conventions)
    # Search in root and common subdirs used by torchvision
    search_paths = [dataset_path]
    tv_folder = 'cifar-100-python' if is_cifar100 else 'cifar-10-batches-py'
    if os.path.isdir(os.path.join(dataset_path, tv_folder)):
         search_paths.append(os.path.join(dataset_path, tv_folder))

    batch_files = []
    test_batch_files = []
    meta_file = None

    for path in search_paths:
        batch_files.extend(glob(os.path.join(path, 'data_batch*')))
        test_batch_files.extend(glob(os.path.join(path, 'test_batch*')))
        potential_meta = os.path.join(path, 'batches.meta' if not is_cifar100 else 'meta')
        if os.path.exists(potential_meta):
             meta_file = potential_meta

    if not batch_files and not test_batch_files:
         log.error(f"Could not find any CIFAR batch files in {dataset_path} or expected subfolders.")
         return {}, None # Return empty structure and no class names

    # Load meta file for class names
    class_names = None
    if meta_file:
         try:
              with open(meta_file, 'rb') as f:
                   meta_data = pickle.load(f, encoding='latin1')
                   class_names = meta_data.get(label_key + '_names') # e.g. fine_label_names or label_names
                   if class_names:
                        log.info(f"Loaded {len(class_names)} class names for {dataset_name} from {meta_file}")
                   else:
                        log.warning(f"Could not find '{label_key}_names' in meta file: {meta_file}")
         except Exception as e:
              log.error(f"Error loading CIFAR meta file {meta_file}: {e}")


    dataset_structure = {'train': [], 'test': []}

    # Load training batches
    for batch_file in sorted(batch_files): # Sort for consistency
        images, labels, _ = load_cifar_batch(batch_file)
        if images is not None:
            for img, label_idx in zip(images, labels):
                # Use fine labels if CIFAR100
                actual_label_idx = label_idx if not is_cifar100 else labels[label_key][label_idx] # Check if labels is dict
                label_name = class_names[actual_label_idx] if class_names and 0 <= actual_label_idx < len(class_names) else f"class_{actual_label_idx}"
                # Store image data directly, not path
                dataset_structure['train'].append((img, label_name))

    # Load test batch
    for batch_file in sorted(test_batch_files):
        images, labels, _ = load_cifar_batch(batch_file)
        if images is not None:
            for img, label_idx in zip(images, labels):
                actual_label_idx = label_idx if not is_cifar100 else labels[label_key][label_idx] # Check if labels is dict
                label_name = class_names[actual_label_idx] if class_names and 0 <= actual_label_idx < len(class_names) else f"class_{actual_label_idx}"
                dataset_structure['test'].append((img, label_name))

    log.info(f"Loaded {len(dataset_structure.get('train',[]))} train and {len(dataset_structure.get('test',[]))} test images for {dataset_name}.")
    return dataset_structure, class_names


# --- MNIST Loader ---

def load_mnist_images_from_file(filepath):
    """Loads MNIST image file (IDX format)."""
    try:
        log.debug(f"Loading MNIST images from: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and dimensions
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            if magic != 2051: raise ValueError(f"Invalid MNIST image file magic number: {magic}")
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            log.debug(f"Image file header: magic={magic}, num_images={num_images}, rows={num_rows}, cols={num_cols}")
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, num_rows, num_cols)
            log.debug(f"Successfully loaded {len(images)} images.")
            return images
    except FileNotFoundError:
        log.error(f"MNIST image file not found: {filepath}")
        return None
    except ValueError as e:
         log.error(f"Error reading MNIST image file header or data {filepath}: {e}")
         return None
    except Exception as e:
        log.error(f"Error loading MNIST images {filepath}: {e}")
        return None

def load_mnist_labels_from_file(filepath):
    """Loads MNIST label file (IDX format)."""
    try:
        log.debug(f"Loading MNIST labels from: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and number of items
            magic = int.from_bytes(f.read(4), 'big')
            num_items = int.from_bytes(f.read(4), 'big')
            if magic != 2049: raise ValueError(f"Invalid MNIST label file magic number: {magic}")
            log.debug(f"Label file header: magic={magic}, num_items={num_items}")
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            log.debug(f"Successfully loaded {len(labels)} labels.")
            return labels
    except FileNotFoundError:
        log.error(f"MNIST label file not found: {filepath}")
        return None
    except ValueError as e:
         log.error(f"Error reading MNIST label file header or data {filepath}: {e}")
         return None
    except Exception as e:
        log.error(f"Error loading MNIST labels {filepath}: {e}")
        return None

def load_mnist_data(dataset_path, dataset_name):
    """Loads MNIST/FashionMNIST data from IDX files."""
    log.info(f"Loading {dataset_name} data from path: {dataset_path}...")
    is_fashion = 'fashion' in dataset_name.lower()

    # Define class names
    if is_fashion:
         class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else: # MNIST
         class_names = [str(i) for i in range(10)]

    # Expected file names (allow variations in naming)
    train_img_patterns = ['train-images-idx3-ubyte.gz', 'train-images.idx3-ubyte']
    train_lbl_patterns = ['train-labels-idx1-ubyte.gz', 'train-labels.idx1-ubyte']
    test_img_patterns = ['t10k-images-idx3-ubyte.gz', 't10k-images.idx3-ubyte']
    test_lbl_patterns = ['t10k-labels-idx1-ubyte.gz', 't10k-labels.idx1-ubyte']

    # Find files, potentially inside subfolders like 'MNIST/raw' or 'FashionMNIST/raw'
    base_paths_to_check = [dataset_path]
    raw_subdir = os.path.join(dataset_path, dataset_name, 'raw') # e.g. MNIST/raw
    if os.path.isdir(raw_subdir): base_paths_to_check.append(raw_subdir)
    root_subdir = os.path.join(dataset_path, dataset_name) # e.g. MNIST/
    if os.path.isdir(root_subdir): base_paths_to_check.append(root_subdir)


    def find_first_existing(patterns, paths):
        for path in paths:
            for pattern in patterns:
                filepath = os.path.join(path, pattern)
                if os.path.exists(filepath):
                    log.debug(f"Found file: {filepath}")
                    return filepath
        log.warning(f"Could not find file matching patterns {patterns} in paths {paths}")
        return None

    train_img_file = find_first_existing(train_img_patterns, base_paths_to_check)
    train_lbl_file = find_first_existing(train_lbl_patterns, base_paths_to_check)
    test_img_file = find_first_existing(test_img_patterns, base_paths_to_check)
    test_lbl_file = find_first_existing(test_lbl_patterns, base_paths_to_check)

    if not train_img_file or not train_lbl_file or not test_img_file or not test_lbl_file:
         log.error(f"Could not find all required MNIST/FashionMNIST IDX files in expected locations within {dataset_path}")
         return {}, class_names # Return empty structure but keep class names

    train_images = load_mnist_images_from_file(train_img_file)
    train_labels = load_mnist_labels_from_file(train_lbl_file)
    test_images = load_mnist_images_from_file(test_img_file)
    test_labels = load_mnist_labels_from_file(test_lbl_file)

    dataset_structure = {'train': [], 'test': []}

    if train_images is not None and train_labels is not None and len(train_images) == len(train_labels):
        for img, label_idx in zip(train_images, train_labels):
            label_name = class_names[label_idx] if class_names and 0 <= label_idx < len(class_names) else f"class_{label_idx}"
            dataset_structure['train'].append((img, label_name)) # Store image data directly

    if test_images is not None and test_labels is not None and len(test_images) == len(test_labels):
        for img, label_idx in zip(test_images, test_labels):
            label_name = class_names[label_idx] if class_names and 0 <= label_idx < len(class_names) else f"class_{label_idx}"
            dataset_structure['test'].append((img, label_name)) # Store image data directly

    log.info(f"Loaded {len(dataset_structure.get('train',[]))} train and {len(dataset_structure.get('test',[]))} test images for {dataset_name}.")
    return dataset_structure, class_names