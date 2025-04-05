import os
import json
import xml.etree.ElementTree as ET
import logging

log = logging.getLogger(__name__)

def parse_voc_xml(annotation_path):
    """Parses a PASCAL VOC XML annotation file."""
    boxes = []
    if not os.path.exists(annotation_path):
        log.error(f"Annotation file not found: {annotation_path}")
        return boxes
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        img_width, img_height = None, None
        size_elem = root.find('size')
        if size_elem is not None:
             width_elem = size_elem.find('width')
             height_elem = size_elem.find('height')
             if width_elem is not None: img_width = int(width_elem.text)
             if height_elem is not None: img_height = int(height_elem.text)


        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            # Handle potential float values in XML, convert safely
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            # Optional: Clamp coordinates to image boundaries if size is known
            if img_width is not None and img_height is not None:
                 xmin = max(0, xmin)
                 ymin = max(0, ymin)
                 xmax = min(img_width - 1, xmax)
                 ymax = min(img_height - 1, ymax)

            boxes.append({'label': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    except ET.ParseError as e:
        log.error(f"Error parsing XML file {annotation_path}: {e}")
    except (ValueError, TypeError) as e:
         log.error(f"Error converting coordinate to int in {annotation_path}: {e}")
    except Exception as e:
        log.error(f"Unexpected error parsing VOC XML {annotation_path}: {e}")
    return boxes

def parse_coco_json(annotation_path, image_filename):
    """
    Parses COCO JSON format (expects a single JSON for the whole split).
    Returns lists of boxes and segmentation data for the given image filename.
    """
    boxes = []
    segments = [] # For segmentation
    if not os.path.exists(annotation_path):
        log.error(f"Annotation file not found: {annotation_path}")
        return boxes, segments

    log.warning(f"COCO JSON parsing ({annotation_path}) is basic. Assumes standard COCO format. Requires pycocotools for full RLE/polygon handling.")
    try:
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        # --- Find image ID ---
        image_id = None
        image_info_found = None
        for img_info in coco_data.get('images', []):
            # Match filename, ignore path components if present in JSON
            if os.path.basename(img_info['file_name']) == image_filename:
                image_id = img_info['id']
                image_info_found = img_info
                log.debug(f"Found image_id {image_id} for filename {image_filename}")
                break

        if image_id is None:
            log.warning(f"Image filename '{image_filename}' not found in COCO JSON {annotation_path}")
            return boxes, segments # Return empty lists

        # --- Create category lookup ---
        category_lookup = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

        # --- Find annotations for this image ID ---
        for ann_info in coco_data.get('annotations', []):
            if ann_info['image_id'] == image_id:
                category_id = ann_info['category_id']
                label = category_lookup.get(category_id, f"cat_{category_id}")

                # Bounding box [xmin, ymin, width, height]
                if 'bbox' in ann_info:
                    bbox = ann_info['bbox']
                    # Ensure coordinates are integers
                    try:
                        xmin, ymin, w, h = map(int, map(float, bbox)) # Allow float conversion first
                        # Clamp coordinates? COCO format might already be valid.
                        boxes.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmin + w, 'ymax': ymin + h})
                    except (ValueError, TypeError) as e:
                         log.warning(f"Skipping invalid bbox {bbox} for image {image_id} in {annotation_path}: {e}")


                # Segmentation (list of polygons [x1,y1,x2,y2,...] or RLE dict)
                if 'segmentation' in ann_info:
                     # Basic handling: store raw data. Drawing requires more complex logic.
                     segments.append({'label': label, 'data': ann_info['segmentation']})

    except json.JSONDecodeError as e:
        log.error(f"Error decoding COCO JSON file {annotation_path}: {e}")
    except Exception as e:
        log.error(f"Unexpected error parsing COCO JSON {annotation_path}: {e}")

    return boxes, segments


def parse_yolo_txt(annotation_path, img_width, img_height):
    """Parses YOLO TXT format (class x_center y_center width height - normalized)."""
    boxes = []
    if not os.path.exists(annotation_path):
        log.error(f"Annotation file not found: {annotation_path}")
        return boxes
    if img_width is None or img_height is None:
         log.error("Image dimensions (width, height) are required to parse YOLO format.")
         return boxes

    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5: # Allow for extra columns sometimes present
                    # Convert safely
                    try:
                        class_id = int(parts[0])
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        width_norm = float(parts[3])
                        height_norm = float(parts[4])
                    except (ValueError, IndexError) as e:
                         log.warning(f"Skipping invalid numeric value in YOLO file {annotation_path}: {line.strip()} - {e}")
                         continue


                    # Convert normalized coordinates to absolute pixel values
                    abs_width = width_norm * img_width
                    abs_height = height_norm * img_height
                    xmin = int((x_center_norm * img_width) - (abs_width / 2))
                    ymin = int((y_center_norm * img_height) - (abs_height / 2))
                    xmax = int(xmin + abs_width)
                    ymax = int(ymin + abs_height)

                    # Clamp coordinates to image boundaries
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width - 1, xmax)
                    ymax = min(img_height - 1, ymax)

                    # Ensure width/height are positive after clamping
                    if xmax <= xmin or ymax <= ymin:
                         log.warning(f"Skipping invalid box dimensions after clamping in YOLO file {annotation_path}: [ {xmin} {ymin} {xmax} {ymax} ] from line {line.strip()}")
                         continue


                    boxes.append({'label': f'class_{class_id}', 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                elif line.strip(): # Log non-empty invalid lines
                    log.warning(f"Skipping invalid line format in YOLO file {annotation_path}: {line.strip()}")
    except Exception as e:
        log.error(f"Error parsing YOLO TXT file {annotation_path}: {e}")
    return boxes

# Add parsers for other formats if needed (e.g., CSV, specific JSON structures)