import torch
import argparse
import sys
import os

# adds project root to sys path --> allows local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detect_faces import detect_faces
from utils.matcher import FaceMatcher
from models.face_classifier import FaceClassifier
from PIL import ImageDraw, ImageFont

def main(image_path, add_name=None, threshold=None):
    """
    Run face detection and recognition on a single image.

    Args:
        image_path (str): Path to the input image.
        add_name (str, optional): If provided, saves the first detected face under this identity.
        threshold (float, optional): Cosine similarity threshold for matching.
    """


    # Detect faces w/ MTCNN
    boxes, image = detect_faces(image_path) # boxes: list of [x1, y1, x2, y2]
    faces = [image.crop(box.tolist()) for box in boxes] # crop face regions from image

    if not faces:
        print("No faces detected.")
        return

    # Load trained, face embedding model (GoogLeNetEmbedder inside FaceClassifier)
    model = FaceClassifier()

    # Load only embedder weights, not classifier head
    checkpoint = torch.load("metrics/best_model_cpu.pth", map_location='cpu')
    model.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith("classifier.")}, strict=False)
    model.eval()

    # Generate 128D embeddings for each face
    embeddings = model.get_embeddings(faces)

    # Load FaceMatcher --> uses cosine similarity
    matcher = FaceMatcher(threshold=threshold)
    matcher.load() # load known embeddings from disk

    # Add new identity if requested
    if add_name:
        print(f"Saving new identity: {add_name}")
        matcher.add(add_name, embeddings[0])  # Add first face only
        matcher.save()
        return

    # Match detected faces against known embeddings
    # Annotate each image w/ boxes and predicted names
    draw = ImageDraw.Draw(image)
    for i, emb in enumerate(embeddings):
        label = matcher.match(emb.cpu())
        print(f"Face {i + 1}: {label}")
        draw.rectangle(boxes[i].tolist(), outline="red", width=2)
        draw.text((boxes[i][0], boxes[i][1] - 10), label, fill="red", font=ImageFont.truetype("/Library/Fonts/Arial.ttf", size=48))

    # Display annotated image
    image.show()

# makes script CLI executable --> python app.py --image image.jpg --add_name name
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--add_name', help='If passed, adds this name as a known identity')
    parser.add_argument('--threshold', type=float, default=0.6) # cosine similarity threshold used by matcher
    args = parser.parse_args()

    main(args.image, add_name=args.add_name, threshold=args.threshold)
