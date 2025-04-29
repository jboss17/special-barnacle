from scripts.detect_faces import detect_faces
from models.face_classifier import FaceClassifier
from utils.matcher import FaceMatcher
from PIL import ImageDraw
import torch

boxes, img = detect_faces("data/sample.jpg", threshold=0.6)
draw = ImageDraw.Draw(img)

for box in boxes:
    draw.rectangle(box.tolist(), outline="red", width=2)

# img.show()

model = FaceClassifier()
checkpoint = torch.load("metrics/best_model_cpu.pth", map_location='cpu')
model.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith("classifier.")}, strict=False)
model.eval()

faces = [img.crop(box.tolist()) for box in boxes]
embeddings = model.get_embeddings(faces)

# for i, emb in enumerate(embeddings):
#     print(f"Face {i}: {emb[:5]}...")  # Preview first 5 dims

# Simulate adding a known face (just reusing one for demo)
matcher = FaceMatcher(threshold=0.65)
matcher.load("data/known_faces.pkl")

if len(embeddings) > 0:
    matcher.add("Person A", embeddings[0])  # First detected face becomes known
    matcher.save("data/known_faces.pkl")

# Test matching all detected faces
for i, emb in enumerate(embeddings):
    name = matcher.match(emb)
    print(f"Face {i} matched: {name}")