import os
import argparse
import logging
import sys

# Try to import from sibling modules
try:
    from .factory import get_downloader, DATASET_CONFIG
    from .url_downloader import UrlDownloader
    from .kaggle_downloader import KaggleDownloader
except ImportError:
    # Handle case where script is run directly without package context
    print("Error: Could not import sibling modules. Ensure you are running this as part of the 'data_retriever' package (e.g., python -m data_retriever ...)")
    # As a fallback for direct script execution (less ideal):
    try:
        from factory import get_downloader, DATASET_CONFIG
        from url_downloader import UrlDownloader
        from kaggle_downloader import KaggleDownloader
    except ImportError as e:
         print(f"Fallback import failed: {e}")
         sys.exit(1)


# --- Constants ---
DEFAULT_OUTPUT_DIR = "./datasets"
# Default download dir relative to output_dir, computed later

# --- Configure Logging ---
# Configure root logger level and formatter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Example: Set specific logger levels if needed
# logging.getLogger('data_retriever.kaggle_downloader').setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize image datasets using predefined configurations or URLs/Kaggle IDs.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting in help text
    )

    # --- Input Source Specification ---
    source_group = parser.add_argument_group('Dataset Source (Choose One)')
    source_exclusive_group = source_group.add_mutually_exclusive_group(required=True)

    # Option 1: Use predefined dataset key
    source_exclusive_group.add_argument(
        "-d", "--dataset-key", type=str,
        help="Key of the predefined dataset to download.\nAvailable keys:\n" + \
             "\n".join([f"  - {k:<20} ({v['type']}, {v['task']})" for k, v in DATASET_CONFIG.items()])
    )

    # Option 2: Specify URL manually
    source_exclusive_group.add_argument(
        "-u", "--url", type=str,
        help="URL of the dataset file (zip, tar.gz, etc.) for generic download."
    )

    # Option 3: Specify Kaggle ID manually
    source_exclusive_group.add_argument(
        "-k", "--kaggle-id", type=str,
        help="Kaggle dataset identifier (e.g., 'user/dataset-name'). Requires Kaggle API setup."
    )

    # --- Options for URL/Kaggle Generic Download ---
    generic_opts = parser.add_argument_group('Options for --url or --kaggle-id')
    generic_opts.add_argument(
        "--generic-task-type", type=str, default="other",
        choices=['classification', 'object_detection', 'segmentation', 'unlabelled', 'other'],
        help="Task type hint for generic download organization (default: other)."
    )
    generic_opts.add_argument(
        "--generic-dataset-name", type=str,
        help="Name for the dataset folder when using --url or --kaggle-id (default: derived from source)."
    )

    # --- Local Path Option ---
    parser.add_argument(
        "-l", "--local-path", type=str,
        help="Path to an already downloaded dataset folder or archive.\n"
             "Required if --dataset-key is of type 'manual'.\n"
             "If provided for other types, uses local data INSTEAD of downloading if the path exists."
    )

    # --- General Options ---
    general_opts = parser.add_argument_group('General Options')
    general_opts.add_argument(
        "-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Base directory to store the final dataset folders (default: {DEFAULT_OUTPUT_DIR})."
    )
    general_opts.add_argument(
        "--download-dir", type=str, # Default computed dynamically based on output-dir
        help="Temporary directory for downloads (default: <output_dir>/downloads_tmp)."
    )
    general_opts.add_argument(
        "--force", action="store_true", default=False,
        help="Force download and organization even if the target dataset directory exists."
    )
    general_opts.add_argument(
        "-v", "--verbose", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO,
        help="Increase output verbosity to DEBUG level."
    )


    args = parser.parse_args()

    # --- Set Log Level ---
    logging.getLogger().setLevel(args.loglevel) # Set root logger level

    # --- Determine Directories ---
    final_output_dir = os.path.abspath(args.output_dir)
    final_download_dir = os.path.abspath(args.download_dir or os.path.join(final_output_dir, "downloads_tmp"))

    # --- Initialize Downloader ---
    downloader = None

    if args.dataset_key:
        logging.debug(f"Using predefined dataset configuration for key: {args.dataset_key}")
        downloader = get_downloader(args.dataset_key, final_output_dir, final_download_dir, args.local_path)
    elif args.url:
        dataset_name = args.generic_dataset_name or args.url.split('/')[-1].split('.')[0].split('?')[0]
        logging.debug(f"Using generic URL downloader for {args.url}")
        # Check if local path exists, if so, use LocalOrganizer instead
        if args.local_path and os.path.exists(os.path.abspath(args.local_path)):
             logging.warning(f"Local path '{args.local_path}' provided and exists. Using local data instead of downloading URL.")
             from .local_organizer import LocalOrganizer # Import here if needed
             downloader = LocalOrganizer(dataset_name, final_output_dir, final_download_dir, args.local_path, args.generic_task_type)
        else:
             downloader = UrlDownloader(dataset_name, final_output_dir, final_download_dir, args.url, args.generic_task_type)
    elif args.kaggle_id:
         dataset_name = args.generic_dataset_name or args.kaggle_id.split('/')[-1]
         logging.debug(f"Using Kaggle downloader for {args.kaggle_id}")
         # Check if local path exists, if so, use LocalOrganizer instead
         if args.local_path and os.path.exists(os.path.abspath(args.local_path)):
              logging.warning(f"Local path '{args.local_path}' provided and exists. Using local data instead of downloading from Kaggle.")
              from .local_organizer import LocalOrganizer # Import here if needed
              downloader = LocalOrganizer(dataset_name, final_output_dir, final_download_dir, args.local_path, args.generic_task_type)
         else:
              downloader = KaggleDownloader(dataset_name, final_output_dir, final_download_dir, args.kaggle_id, args.generic_task_type)
    # else: # Should be caught by argparse mutually exclusive group requirement

    # --- Execute Retrieval ---
    if downloader:
        success = downloader.retrieve(force=args.force)
        if success:
            logging.info(f"Dataset retrieval process completed for '{downloader.dataset_name}'.")
            print(f"\nDataset ready at: {downloader.dataset_final_path}") # Print final path clearly
            sys.exit(0)
        else:
            logging.error(f"Dataset retrieval process failed for '{downloader.dataset_name}'.")
            sys.exit(1)
    else:
        # Error should have been logged by get_downloader or argparse
        logging.critical("Could not initialize a suitable downloader based on the provided arguments.")
        sys.exit(1)

if __name__ == "__main__":
    main()