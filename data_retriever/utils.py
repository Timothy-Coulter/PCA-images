import os
import requests
import zipfile
import tarfile
import shutil
import logging
from tqdm import tqdm

# Configure logging (can be configured globally later)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use module-specific logger

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

            log.info(f"Downloading {local_filename} from {url}...")
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=local_filename, leave=False)
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                log.error("ERROR, download size mismatch.")
                # Keep partial file for inspection? Or remove? For now, keep.
                # if os.path.exists(local_filepath): os.remove(local_filepath)
                # return None
            log.info(f"Successfully downloaded {local_filename}")
            return local_filepath
    except requests.exceptions.RequestException as e:
        log.error(f"Error downloading {url}: {e}")
        if os.path.exists(local_filepath): os.remove(local_filepath)
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred during download: {e}")
        if os.path.exists(local_filepath): os.remove(local_filepath)
        return None

def extract_archive(filepath, destination_folder, remove_archive=True):
    """Extracts zip or tar archives."""
    if not os.path.exists(filepath):
        log.error(f"Archive file not found: {filepath}")
        return False, None

    log.info(f"Extracting {os.path.basename(filepath)} to {destination_folder}...")
    os.makedirs(destination_folder, exist_ok=True)
    extracted_path = destination_folder # Default if no single top-level folder

    try:
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                # Check for single top-level directory
                top_level_dirs = list(set(item.split('/')[0] for item in zip_ref.namelist() if '/' in item)) # Ensure there's a separator
                if len(top_level_dirs) == 1:
                    extracted_path = os.path.join(destination_folder, top_level_dirs[0])
                zip_ref.extractall(destination_folder)
            log.info("ZIP extraction complete.")
        elif tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r:*') as tar_ref:
                 # Check for single top-level directory
                members = tar_ref.getmembers()
                top_level_dirs = list(set(m.name.split('/')[0] for m in members if m.isdir() and '/' not in m.name.strip('/'))) # Check top-level dirs
                # More robust check: find common prefix if needed
                if len(top_level_dirs) == 1:
                     extracted_path = os.path.join(destination_folder, top_level_dirs[0])
                tar_ref.extractall(path=destination_folder)
            log.info("TAR extraction complete.")
        else:
            log.warning(f"File {os.path.basename(filepath)} is not a recognized archive format (zip/tar). Skipping extraction.")
            return False, filepath # Return original path if not archive

        if remove_archive:
            os.remove(filepath)
            log.info(f"Removed archive file: {os.path.basename(filepath)}")
        return True, extracted_path # Indicate success and path to extracted content
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e: # Added EOFError
        log.error(f"Error extracting archive {os.path.basename(filepath)}: {e}")
        return False, None
    except Exception as e:
        log.error(f"An unexpected error occurred during extraction: {e}")
        return False, None