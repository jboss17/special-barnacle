import torch
from PIL import Image # loading/handling images
from facenet_pytorch import MTCNN # face detector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN detector --> return all detected faces
mtcnn = MTCNN(keep_all=True, device=device)
#threshold --> confidence level to filter weak detections
def detect_faces(image_path, threshold=0.9):
    """
    Detect faces using MTCNN.

    Args:
        image_path (str): Path to the input image.
        threshold (float): Confidence threshold for face detection.

    Returns:
        boxes (List[Tensor]): List of bounding boxes [x1, y1, x2, y2]. 
        image (PIL.Image): Original input image for later drawing + cropping.
    """
    image = Image.open(image_path).convert("RGB")

    # Detect faces
    # --> boxes: bounding boxes for each detected face
    # --> probs: detected confidence for each face
    boxes, probs = mtcnn.detect(image)

    # Filter by confidence threshold --> any boxes below our threshold are gone
    filtered_boxes = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is not None and prob > threshold:
                box = [float(coord) for coord in box]
                filtered_boxes.append(torch.tensor(box))

    # face_crops = mtcnn.extract(image, filtered_boxes, save_path=None)

    return filtered_boxes, image