import os
import torch
from PIL import ImageDraw
import matplotlib.pyplot as plt

from scripts.detect_faces import detect_faces
from models.face_classifier import FaceClassifier
from utils.matcher import FaceMatcher

# Load FDDB image paths --> recursively gather all .jpg files in FDDB dataset
fddb_root = 'data/Dataset_FDDB/images'
fddb_imgs = []
for root, _, files in os.walk(fddb_root):
    for file in files:
        if file.endswith('.jpg'):
            fddb_imgs.append(os.path.join(root, file))

# Load trained embedding model
model = FaceClassifier()

# Load only the embedder (ignore classifier layer)
checkpoint = torch.load("metrics/best_model_cpu.pth", map_location='cpu')
model.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith("classifier.")}, strict=False)
model.eval()

# Load FaceMatcher with known identities
matcher = FaceMatcher()
matcher.load("data/known_faces.pkl")

print(f"Running on {len(fddb_imgs[:5])} FDDB images...\n")

# Process the first 5 images for testing and visualize results
for path in fddb_imgs[:5]:
    # Detect faces and bounding boxes
    boxes, img = detect_faces(path, threshold=0.7)

    # Crop faces manually using the bounding boxes
    faces = [img.crop(box.tolist()) for box in boxes]

    # Generate 128D embeddings for each cropped face
    embeddings = model.get_embeddings(faces)

    # Initialize drawing context
    draw = ImageDraw.Draw(img)

    print(f"\n{os.path.basename(path)} - Found {len(boxes)} face(s):")

    # Match each embedding to a known identity and annotate
    for i, (emb, box) in enumerate(zip(embeddings, boxes)):
        name = matcher.match(emb)
        print(f"  Face {i + 1}: {name}")
        draw.rectangle(box.tolist(), outline="lime", width=2)
        draw.text((box[0], box[1] - 10), name, fill="lime")

    # Show image with annotations
    plt.imshow(img)
    plt.title(f"{os.path.basename(path)} — Detected: {len(boxes)}")
    plt.axis("off")
    plt.show()
