import argparse
import logging
import sys
import matplotlib.pyplot as plt

# Try to import from sibling modules
try:
    from .visualiser import DatasetVisualiser
except ImportError:
    # Handle case where script is run directly without package context
    print("Error: Could not import sibling modules. Ensure you are running this as part of the 'data_visualiser' package (e.g., python -m data_visualiser ...)")
    # As a fallback for direct script execution (less ideal):
    try:
        from visualiser import DatasetVisualiser
    except ImportError as e:
         print(f"Fallback import failed: {e}")
         sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(
        description="Visualize image datasets organized by the data_retriever tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("dataset_path", type=str, help="Path to the root directory of the organized dataset.")
    parser.add_argument("-t", "--task-type", type=str, required=True,
                        choices=['classification', 'object_detection', 'segmentation', 'unlabelled', 'other'],
                        help="Type of computer vision task the dataset is for. This helps in loading and displaying annotations correctly.")
    parser.add_argument("--dataset-name", type=str,
                        help="Optional name hint for the dataset (e.g., 'cifar10', 'mnist'). Useful if the dataset uses specific file formats (like CIFAR batches or MNIST IDX files) instead of standard image folders.")

    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument("--view-samples", action=argparse.BooleanOptionalAction, default=True,
                           help="View sample images (default: True). Use --no-view-samples to disable.")
    vis_group.add_argument("--run-eda", action=argparse.BooleanOptionalAction, default=True,
                           help="Run basic EDA (e.g., class distribution plot) (default: True). Use --no-run-eda to disable.")
    vis_group.add_argument("-n", "--num-samples", type=int, default=9,
                           help="Number of samples to visualize.")
    vis_group.add_argument("--cols", type=int, default=3,
                           help="Number of columns in the sample visualization grid.")
    vis_group.add_argument("--splits", nargs='+', default=None,
                           help="Specific splits to visualize (e.g., 'train' 'test'). Default is all available splits.")
    vis_group.add_argument("--hide-plots", action="store_true", default=False,
                           help="Generate plots but do not display them interactively (useful for testing).")

    parser.add_argument(
        "-v", "--verbose", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO,
        help="Increase output verbosity to DEBUG level."
    )

    args = parser.parse_args()

    # --- Set Log Level ---
    logging.getLogger().setLevel(args.loglevel) # Set root logger level
    # Set matplotlib log level higher to avoid excessive debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


    # --- Run Visualizer ---
    try:
        visualiser = DatasetVisualiser(args.dataset_path, args.task_type, args.dataset_name)

        # Check if data was loaded
        if not visualiser.dataset_structure or all(not items for items in visualiser.dataset_structure.values()):
             logging.error("Visualiser failed to load any data from the specified path and dataset type.")
             sys.exit(1)

        show_plots_flag = not args.hide_plots

        if args.run_eda:
            visualiser.run_eda(show_plots=show_plots_flag)
            # If only running EDA and not viewing samples, and plots are hidden, we might exit early.
            # However, let's allow viewing samples even if EDA plot is hidden.
            # If showing plots for EDA, plt.show() might be called within run_eda if implemented that way,
            # or we might need to call it here if run_eda only generates the figure.
            # Current implementation: run_eda generates plot if show_plots=True, but doesn't call plt.show().

        if args.view_samples:
            visualiser.view_samples(num_samples=args.num_samples, cols=args.cols, splits=args.splits, show_plots=show_plots_flag)
        elif show_plots_flag and args.run_eda:
             # If we ran EDA with show_plots=True but didn't view samples,
             # we need to explicitly show the EDA plot now.
             if plt.get_fignums(): # Check if any figures were created
                  logging.debug("Displaying generated EDA plot window...")
                  plt.show()
             else:
                  logging.debug("EDA ran, but no plots were generated to display.")


        logging.info("Visualization process finished.")
        sys.exit(0)

    except FileNotFoundError as e:
        logging.critical(e) # Use critical for fatal errors like file not found
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()