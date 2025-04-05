import os
import logging
from glob import glob

log = logging.getLogger(__name__)

def find_image_files(directory):
    """Finds common image files recursively in a directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    log.debug(f"Searching for images in: {directory}")
    for ext in image_extensions:
        # Search in the directory itself and one level down (common for split/class structure)
        image_files.extend(glob(os.path.join(directory, ext)))
        image_files.extend(glob(os.path.join(directory, '*', ext)))
        # Add recursive search if needed, but can be slow on large datasets
        # image_files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
    log.debug(f"Found {len(image_files)} potential image files.")
    # Filter out potential duplicates if recursive search is added
    return sorted(list(set(image_files)))


def find_annotation_file(img_path, annotation_dir, expected_exts):
    """
    Tries to find a matching annotation file for an image in the annotation directory.
    Matches based on filename (excluding extension).
    """
    if not os.path.isdir(annotation_dir):
        log.warning(f"Annotation directory does not exist: {annotation_dir}")
        return None

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    log.debug(f"Searching for annotation for image base name: {img_basename} in {annotation_dir}")

    for annot_ext in expected_exts:
        # Check directly in the annotation dir
        potential_annot_path = os.path.join(annotation_dir, img_basename + annot_ext)
        if os.path.exists(potential_annot_path):
            log.debug(f"Found direct annotation match: {potential_annot_path}")
            return potential_annot_path

    # If not found directly, search recursively (might be needed for complex structures like VOC)
    # Be cautious with performance on very large/deep annotation directories.
    log.debug(f"Direct annotation match not found, searching recursively in {annotation_dir}...")
    for root, _, files in os.walk(annotation_dir):
         for fname in files:
              if os.path.splitext(fname)[0] == img_basename:
                   # Check if the extension matches expected ones
                   if any(fname.endswith(ext) for ext in expected_exts):
                        found_path = os.path.join(root, fname)
                        log.debug(f"Found recursive annotation match: {found_path}")
                        return found_path

    log.debug(f"No annotation found for {img_basename} with extensions {expected_exts} in {annotation_dir}")
    return None