import os
import csv
from detect_faces import detect_faces
from PIL import Image
from pathlib import Path

# Define paths
fddb_root = 'data/Dataset_FDDB/images'
output_dir = 'data/fddb_crops'
os.makedirs(output_dir, exist_ok=True)

# Collect all FDDB images
fddb_imgs = []
for root, _, files in os.walk(fddb_root):
    for file in files:
        if file.endswith('.jpg'):
            fddb_imgs.append(os.path.join(root, file))

print(f"[INFO] Found {len(fddb_imgs)} FDDB images.")

# Prep labels CSV file
labels_path = os.path.join(output_dir, 'fddb_labels.csv')
with open(labels_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])

    # Process each image
    for img_path in fddb_imgs:
        img_id = Path(img_path).stem.replace('img_', '')  # use image number as label

       # Detect faces in image w/ MTCNN
        boxes, image = detect_faces(img_path, threshold=0.7)

        # If no faces detected --> skip
        if boxes is None or len(boxes) == 0:
            continue

        # Crop + save each detected face
        for i, box in enumerate(boxes):
            box = [int(float(coord)) for coord in box] # ensure coordinates are ints
            cropped = image.crop(box)

            # Save cropped face image
            crop_filename = f"{img_id}_face{i}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cropped.save(crop_path)

            # Write filename and label to CSV
            writer.writerow([crop_filename, img_id])
