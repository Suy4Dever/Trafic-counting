import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil
import random

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Where you want the output to go
OUTPUT_PATH = os.path.join(BASE_DIR, 'UA-DETRAC-Processed')

# Point these to your downloaded folders
# (Adjust these names if yours are slightly different)
IMAGES_DIR_ROOT = os.path.join(BASE_DIR, 'DETRAC-Images')
ANNOTATIONS_DIR_ROOT = os.path.join(BASE_DIR, 'DETRAC-Train-Annotations-XML')

CLASS_MAPPING = {'car': 0, 'bus': 1, 'van': 2, 'others': 3}

def find_actual_folder(start_path, target_suffix):
    """Deep search to find where the actual XMLs or MVI folders are."""
    for root, dirs, files in os.walk(start_path):
        # If we find XML files, this is our annotation folder
        if any(f.endswith('.xml') for f in files):
            return root
        # If we find folders starting with MVI, this is our image root
        if any(d.startswith('MVI_') for d in dirs):
            return root
    return None

def process_detrac_annotations():
    # 1. Find the real folders automatically
    print("🔍 Searching for data folders...")
    real_xml_dir = find_actual_folder(ANNOTATIONS_DIR_ROOT, '.xml')
    real_img_dir = find_actual_folder(IMAGES_DIR_ROOT, 'MVI_')

    if not real_xml_dir or not real_img_dir:
        print(f"❌ ERROR: Could not find data.")
        print(f"Looked in: {IMAGES_DIR_ROOT} and {ANNOTATIONS_DIR_ROOT}")
        print("Make sure you unzipped the files into these folders!")
        return

    print(f"✅ Found XMLs in: {real_xml_dir}")
    print(f"✅ Found Images in: {real_img_dir}")

    # 2. Create Output Structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_PATH, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, f'labels/{split}'), exist_ok=True)

    # 3. Process
    xml_files = [f for f in os.listdir(real_xml_dir) if f.endswith('.xml')]
    processed_count = 0

    for xml_file in tqdm(xml_files, desc="Processing Sequences"):
        sequence_name = os.path.splitext(xml_file)[0]
        image_sequence_dir = os.path.join(real_img_dir, sequence_name)

        if not os.path.isdir(image_sequence_dir):
            continue

        tree = ET.parse(os.path.join(real_xml_dir, xml_file))
        root = tree.getroot()

        # Get dims from first image
        try:
            sample_img = os.path.join(image_sequence_dir, sorted(os.listdir(image_sequence_dir))[0])
            with Image.open(sample_img) as img:
                w_full, h_full = img.size
        except: continue

        for frame in root.findall('frame'):
            frame_num = int(frame.get('num'))
            img_name = f"img{frame_num:05d}.jpg"
            src_path = os.path.join(image_sequence_dir, img_name)

            if not os.path.exists(src_path): continue

            # Split 80/20
            split = 'train' if random.random() > 0.2 else 'val'
            unique_name = f"{sequence_name}_img{frame_num:05d}"
            
            dest_img = os.path.join(OUTPUT_PATH, f'images/{split}', f"{unique_name}.jpg")
            dest_txt = os.path.join(OUTPUT_PATH, f'labels/{split}', f"{unique_name}.txt")

            shutil.copy(src_path, dest_img)

            yolo_data = []
            target_list = frame.find('target_list')
            if target_list is not None:
                for target in target_list.findall('target'):
                    box = target.find('box')
                    attr = target.find('attribute')
                    v_type = attr.get('vehicle_type')
                    
                    if v_type in CLASS_MAPPING:
                        cid = CLASS_MAPPING[v_type]
                        # Calc YOLO coords
                        bw = float(box.get('width'))
                        bh = float(box.get('height'))
                        bx = float(box.get('left')) + bw/2
                        by = float(box.get('top')) + bh/2
                        
                        yolo_data.append(f"{cid} {bx/w_full:.6f} {by/h_full:.6f} {bw/w_full:.6f} {bh/h_full:.6f}")

            with open(dest_txt, 'w') as f:
                f.write('\n'.join(yolo_data))
            processed_count += 1

    print(f"\n🚀 SUCCESS! Processed {processed_count} frames.")
    print(f"Your data is ready in: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_detrac_annotations()