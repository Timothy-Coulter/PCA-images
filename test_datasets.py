import os
import argparse
import logging
import sys
import matplotlib.pyplot as plt

# Add the directory containing the scripts to the Python path
# This assumes test_datasets.py is in the same directory as the other two scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# No longer need to modify sys.path if running as modules or test script is at root

try:
    # Import from the new package structure
    from data_retriever.factory import get_downloader, DATASET_CONFIG
    # Define default output dir here or import if needed, assuming test script is at root relative to packages
    DEFAULT_OUTPUT_DIR = "./datasets"
    from data_visualiser.visualiser import DatasetVisualiser
except ImportError as e:
    print(f"Error importing required modules from packages: {e}")
    print("Please ensure 'data_retriever' and 'data_visualiser' packages are accessible (e.g., in the same parent directory or installed).")
    sys.exit(1)

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [TEST] %(message)s')

# --- Datasets to Test ---
# Map user-friendly names or common IDs to the keys used in DataRetriever.DATASET_CONFIG
# Add more mappings as needed
TEST_DATASET_KEYS = {
    # Small
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "voc2007_detect": "voc2007_detect",
    "voc2007_segment": "voc2007_segment",
    "openimages_subset": "openimages_subset", # Manual
    "camvid": "camvid", # URL

    # Medium
    "oxford_pets": "oxford_pets",
    "flowers102": "flowers102",
    "stanford_dogs": "stanford_dogs",
    "coco2017_detect": "coco2017_detect", # Manual
    "coco2017_segment": "coco2017_segment", # Manual
    "kitti_detect": "kitti_detect", # Manual
    "ade20k": "ade20k", # URL
    "cityscapes": "cityscapes", # Manual
}

MANUAL_DATASETS = {k for k, v in DATASET_CONFIG.items() if v['type'] == 'manual'}

def parse_local_paths(path_args):
    """Parses --local-paths arguments into a dictionary."""
    local_paths_dict = {}
    if path_args:
        for path_arg in path_args:
            parts = path_arg.split(':', 1)
            if len(parts) == 2:
                key, path = parts
                if key in DATASET_CONFIG:
                    local_paths_dict[key] = os.path.abspath(path)
                else:
                    logging.warning(f"Invalid dataset key '{key}' in --local-paths argument: {path_arg}. Skipping.")
            else:
                logging.warning(f"Invalid format for --local-paths argument: {path_arg}. Expected format: 'dataset_key:/path/to/data'. Skipping.")
    return local_paths_dict

def run_test(dataset_key, output_dir, download_dir, local_paths, show_plots):
    """Runs the download and visualization test for a single dataset."""
    logging.info(f"--- Testing Dataset: {dataset_key} ---")
    test_success = False
    retrieved_path = None
    task_type = DATASET_CONFIG.get(dataset_key, {}).get('task', 'other') # Get task type from config

    # --- Step 1: Retrieve Dataset ---
    try:
        local_path_for_key = local_paths.get(dataset_key)
        if dataset_key in MANUAL_DATASETS and not local_path_for_key:
            logging.warning(f"Skipping manual dataset '{dataset_key}' because --local-paths was not provided for it.")
            return False # Consider skipped tests as failure for overall status? Or track separately? Let's say failure.

        downloader = get_downloader(dataset_key, output_dir=output_dir, download_dir=download_dir, local_path=local_path_for_key)

        if not downloader:
            logging.error(f"Failed to get downloader for '{dataset_key}'.")
            return False

        # Check if dataset already exists (downloader.retrieve handles this but we can log it)
        if os.path.exists(downloader.dataset_final_path):
             logging.info(f"Dataset directory already exists: {downloader.dataset_final_path}")
             retrieved_path = downloader.dataset_final_path # Use existing path
             retrieval_success = True
        else:
             logging.info(f"Attempting retrieval for '{dataset_key}'...")
             retrieval_success = downloader.retrieve()
             if retrieval_success:
                  retrieved_path = downloader.dataset_final_path
                  logging.info(f"Retrieval successful. Dataset at: {retrieved_path}")
             else:
                  logging.error(f"Retrieval failed for '{dataset_key}'.")
                  return False

    except Exception as e:
        logging.error(f"Error during retrieval phase for '{dataset_key}': {e}", exc_info=True)
        return False

    # --- Step 2: Visualize Dataset ---
    if retrieved_path and os.path.exists(retrieved_path):
        try:
            logging.info(f"Attempting visualization for '{dataset_key}' at {retrieved_path}...")
            # Use dataset_key as dataset_name hint for visualiser if needed (e.g., for CIFAR/MNIST)
            visualiser = DatasetVisualiser(retrieved_path, task_type=task_type, dataset_name=dataset_key)

            # Check if data was loaded
            if not visualiser.dataset_structure:
                 logging.error(f"Visualiser failed to load data structure for '{dataset_key}'.")
                 return False # Failed visualization step

            logging.info(f"Running EDA for '{dataset_key}'...")
            visualiser.run_eda() # Run EDA (plots might show if show_plots is True)

            logging.info(f"Visualizing 1 sample for '{dataset_key}'...")
            visualiser.view_samples(num_samples=1, cols=1)

            if show_plots:
                 logging.info("Displaying plots (test might block here)...")
                 plt.show() # This will block if show_plots is True
            else:
                 plt.close('all') # Close figures immediately if not showing

            logging.info(f"Visualization check completed for '{dataset_key}'.")
            test_success = True # Both retrieval and visualization steps seemed okay

        except FileNotFoundError as e:
             logging.error(f"Visualization Error (FileNotFound) for '{dataset_key}': {e}")
        except ImportError as e:
             logging.error(f"Visualization Error (ImportError, maybe missing libraries like matplotlib?) for '{dataset_key}': {e}")
        except Exception as e:
            logging.error(f"Error during visualization phase for '{dataset_key}': {e}", exc_info=True)
            # Don't necessarily fail the whole test if visualization fails, but log it.
            # Let's consider visualization failure as test failure for now.
            test_success = False
    else:
        logging.error(f"Retrieved path '{retrieved_path}' not found or invalid after retrieval step for '{dataset_key}'.")
        test_success = False


    logging.info(f"--- Test Result for {dataset_key}: {'SUCCESS' if test_success else 'FAILURE'} ---")
    return test_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for DataRetriever and DatasetVisualiser.")

    parser.add_argument("--datasets", nargs='+', default=list(TEST_DATASET_KEYS.keys()),
                        help=f"Space-separated list of dataset keys/names to test (default: all known). Available: {list(TEST_DATASET_KEYS.keys())}")
    parser.add_argument("--local-paths", nargs='+',
                        help="Provide paths for MANUALLY downloaded datasets. Format: 'dataset_key:/path/to/data' (e.g., 'coco2017_detect:/data/coco')")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Base directory to store downloaded datasets (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--download-dir", type=str, # Default computed dynamically
                        help="Temporary directory for downloads (default: <output_dir>/downloads_tmp).")
    parser.add_argument("--show-plots", action="store_true", default=False,
                        help="Show matplotlib plots during visualization (can block execution). Default is False.")

    args = parser.parse_args()

    local_paths_dict = parse_local_paths(args.local_paths)
    final_output_dir = os.path.abspath(args.output_dir)
    final_download_dir = os.path.abspath(args.download_dir or os.path.join(final_output_dir, "downloads_tmp"))

    # Validate requested datasets
    datasets_to_test = []
    invalid_keys = []
    for key_or_name in args.datasets:
        if key_or_name in TEST_DATASET_KEYS:
            datasets_to_test.append(TEST_DATASET_KEYS[key_or_name])
        elif key_or_name in DATASET_CONFIG: # Allow direct use of config keys
             datasets_to_test.append(key_or_name)
        else:
            invalid_keys.append(key_or_name)

    if invalid_keys:
        logging.warning(f"Ignoring invalid dataset keys/names: {invalid_keys}")

    if not datasets_to_test:
        logging.error("No valid datasets selected to test.")
        sys.exit(1)

    logging.info(f"Starting tests for datasets: {datasets_to_test}")
    logging.info(f"Using output directory: {final_output_dir}")
    logging.info(f"Using temporary download directory: {final_download_dir}")
    if local_paths_dict:
        logging.info(f"Using local paths: {local_paths_dict}")
    if not args.show_plots:
         logging.info("Plots will be generated but not displayed interactively (--show-plots not set).")


    results = {}
    overall_success = True

    for dataset_key in datasets_to_test:
        success = run_test(dataset_key, final_output_dir, final_download_dir, local_paths_dict, args.show_plots)
        results[dataset_key] = success
        if not success:
            overall_success = False

    # --- Summary ---
    logging.info("--- Test Summary ---")
    for key, success in results.items():
        logging.info(f"Dataset: {key:<20} | Result: {'SUCCESS' if success else 'FAILURE'}")
    logging.info("--------------------")

    if overall_success:
        logging.info("All specified tests passed successfully!")
        sys.exit(0)
    else:
        logging.error("One or more tests failed.")
        sys.exit(1)