import os
import logging
from .torchvision_downloader import TorchvisionDownloader
from .url_downloader import UrlDownloader
from .kaggle_downloader import KaggleDownloader
from .local_organizer import LocalOrganizer

log = logging.getLogger(__name__)

# --- Dataset Registry ---
# Moved from the original script
DATASET_CONFIG = {
    # Small datasets
    "cifar10": {"type": "torchvision", "id": "CIFAR10", "task": "classification"},
    "cifar100": {"type": "torchvision", "id": "CIFAR100", "task": "classification"},
    "mnist": {"type": "torchvision", "id": "MNIST", "task": "classification"},
    "fashion_mnist": {"type": "torchvision", "id": "FashionMNIST", "task": "classification"},
    "voc2007_detect": {"type": "torchvision", "id": "VOCDetection", "task": "object_detection", "kwargs": {"year": "2007", "image_set": "trainval"}},
    "voc2007_segment": {"type": "torchvision", "id": "VOCSegmentation", "task": "segmentation", "kwargs": {"year": "2007", "image_set": "trainval"}},
    "camvid": {"type": "url", "id": "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip", "task": "segmentation", "note": "Check URL. Structure needs specific organizer."},
    "openimages_subset": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download or specific tools (e.g., FiftyOne). Provide path via --local-path."},

    # Medium datasets
    "oxford_pets": {"type": "torchvision", "id": "OxfordIIITPet", "task": "segmentation"}, # Also classification
    "flowers102": {"type": "torchvision", "id": "Flowers102", "task": "classification"},
    "stanford_dogs": {"type": "torchvision", "id": "StanfordDogs", "task": "classification"},
    "coco2017_detect": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download of images (train/val) and annotations (train/val). Provide path via --local-path."},
    "coco2017_segment": {"type": "manual", "id": None, "task": "segmentation", "note": "Requires manual download of images (train/val) and annotations (train/val). Provide path via --local-path."},
    "kitti_detect": {"type": "manual", "id": None, "task": "object_detection", "note": "Requires manual download from KITTI website. Provide path via --local-path."},
    "ade20k": {"type": "url", "id": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip", "task": "segmentation", "note": "Check URL. Structure needs specific organizer."},
    "cityscapes": {"type": "manual", "id": None, "task": "segmentation", "note": "Requires registration and manual download from Cityscapes website (gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip). Provide path via --local-path."},

    # Add other datasets here
}

# --- Factory Function ---

def get_downloader(dataset_key, output_dir, download_dir, local_path=None):
    """
    Factory function to get the appropriate downloader/organizer instance based on config or local path.

    Args:
        dataset_key (str): The key identifying the dataset in DATASET_CONFIG.
        output_dir (str): Base directory for final organized datasets.
        download_dir (str): Base directory for temporary downloads.
        local_path (str, optional): Path to already downloaded data. If provided and exists,
                                     LocalOrganizer is used instead of downloading. Required for
                                     datasets marked as 'manual'. Defaults to None.

    Returns:
        BaseDownloader or None: An instance of a downloader/organizer class, or None if config is invalid
                                or required local path is missing/invalid.
    """
    if dataset_key not in DATASET_CONFIG:
        log.error(f"Unknown dataset key: '{dataset_key}'. Available keys: {list(DATASET_CONFIG.keys())}")
        return None

    config = DATASET_CONFIG[dataset_key]
    dataset_name = dataset_key # Use the key as the folder name by default
    dataset_type = config["type"]
    dataset_id = config.get("id") # Might be None for manual
    task_type = config["task"]
    kwargs = config.get("kwargs", {})

    # --- Resolve Local Path ---
    resolved_local_path = None
    if local_path:
        abs_local_path = os.path.abspath(local_path)
        if os.path.exists(abs_local_path):
            resolved_local_path = abs_local_path
            log.info(f"Validated local path exists: {resolved_local_path}")
        else:
            # If path provided but doesn't exist, it's an error only if dataset is manual
            if dataset_type == "manual":
                log.error(f"Dataset '{dataset_key}' is manual, but required local path '{abs_local_path}' not found.")
                return None
            else:
                # For non-manual, just warn and proceed without local path
                log.warning(f"Provided local path '{abs_local_path}' not found. Ignoring and attempting download.")
                resolved_local_path = None # Ensure it's None

    # --- Instantiate based on type and local path presence ---

    # Priority 1: Use LocalOrganizer if local path is valid (required for manual, optional override for others)
    if resolved_local_path:
        if dataset_type == "manual":
            log.info(f"Using LocalOrganizer for manual dataset '{dataset_key}' with path: {resolved_local_path}")
        else:
            log.warning(f"Using LocalOrganizer for non-manual dataset '{dataset_key}' due to provided local path: {resolved_local_path}. Download will be skipped.")
        return LocalOrganizer(dataset_name, output_dir, download_dir, resolved_local_path, task_type)

    # Priority 2: If no valid local path, proceed with configured download type
    # (Error if manual type reached here, as local path was required but invalid/missing)
    if dataset_type == "manual":
         log.error(f"Dataset '{dataset_key}' is manual, but a valid local path was not provided or found.")
         return None
    elif dataset_type == "torchvision":
        log.info(f"Using TorchvisionDownloader for '{dataset_key}'.")
        return TorchvisionDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type, **kwargs)
    elif dataset_type == "url":
        log.info(f"Using UrlDownloader for '{dataset_key}'.")
        # TODO: Add specific URL organizers here if needed, e.g., for CamVid, ADE20k
        # if dataset_key == 'camvid': return CamVidOrganizer(...)
        return UrlDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type)
    elif dataset_type == "kaggle":
        log.info(f"Using KaggleDownloader for '{dataset_key}'.")
        return KaggleDownloader(dataset_name, output_dir, download_dir, dataset_id, task_type)
    else:
        log.error(f"Unsupported dataset type '{dataset_type}' in config for '{dataset_key}'.")
        return None