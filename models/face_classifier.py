import torch
import torch.nn as nn
import torch.nn.functional as F
from recognize_faces import GoogLeNetEmbedder
from torchvision import transforms

# Image Preprocessing
#   - Resize to 128x128
#   - Normalize ot [-1, 1] range
crop_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class FaceClassifier(nn.Module):
    """
    A face classification model using GoogLeNet as feature extractor.

    Can output either:
    - Class logits for classification (during training)
    - 128D face embeddings (for recognition/matching)

    Args:
        num_classes (int, optional): Number of target identities (for classification).
        embedding_dim (int): Dimensionality of the face embedding vector.
    """
    def __init__(self, num_classes=None, embedding_dim=128):
        super(FaceClassifier, self).__init__()

        # Extracts 128D feature vector from each face
        self.embedder = GoogLeNetEmbedder(embedding_dim=embedding_dim)

        # Only create classifier head if num_classes is provided
        self.has_classifier = num_classes is not None
        if self.has_classifier:
            # Linear layer that maps 128D vectors to num_classes logits
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input batch of face images [N, 3, H, W].

        Returns:
            Tensor: Class logits [N, num_classes] if classifier exists,
                    otherwise embeddings [N, embedding_dim]. This is used in CrossEntropyLoss
        """
        embeddings = self.emberdder(x)

        # If classifier exists --> output logits else return embeddings
        if self.has_classifier:
            logits = self.classifier(embeddings)
            return logits
        else:
            return embeddings

    def get_embeddings(self, face_crops, device='cpu'):
        """
        Preprocess face crops and extract 128D embeddings.

        Args:
            face_crops (list[PIL.Image]): List of cropped face images.
            device (str): Device to run the model ('cpu' or 'cuda').

        Returns:
            Tensor: Batch of embeddings with shape [N, 128].
        """
        self.eval()

        # Preprocess faces: resize, normalize, add batch dimension
        tensors = [crop_transform(face).unsqueeze(0).to(device) for face in face_crops]
        batch = torch.cat(tensors, dim=0)  # Shape: [N, 3, H, W]

        # Generate embeddings without gradient tracking
        with torch.no_grad():
            embeddings = self.embedder(batch)

        return embeddings.cpu()