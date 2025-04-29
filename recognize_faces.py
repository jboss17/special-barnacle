import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class GoogLeNetEmbedder(nn.Module):
    """
    A face embedding model that uses GoogLeNet as the feature extractor.
    Extracts 128D vectors from input face images.

    Attributes:
        backbone (nn.Module): GoogLeNet model without classification head.
        embed (nn.Linear): Linear layer to reduce GoogLeNet output to 128D.
    """
    def __init__(self, embedding_dim=128):
        """
        Initialize the embedder with GoogLeNet backbone and 128D projection layer.

        Args:
            embedding_dim (int): Dimensionality of the final face embedding.
        """
        super().__init__()

        # Load pretrained GoogLeNet with auxiliary classifiers enabled
        #   This helps teh model train better --> provides extra gradient signals early in network
        base = models.googlenet(weights='DEFAULT', aux_logits=True)

        # Remove classification head (raw feature vectors only)
        base.fc = nn.Identity()
        self.backbone = base

        # Final layer to reduce 1024D GoogLeNet features to desired embedding_dim (default 128)
        self.embed = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        """
        Forward pass: extract 128D normalized embeddings from input image batch.

        Args:
            x (Tensor): Input tensor of shape [N, 3, H, W]

        Returns:
            Tensor: Normalized face embeddings of shape [N, embedding_dim]
        """

        # Pass input through GoogLeNet backbone
        out = self.backbone(x)

        # Handle case where GoogLeNet returns (main_output, aux1, aux2)
        #   train() --> tuple
        #   eval() --> single tensor
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out

        # Project to 128D embedding space
        x = self.embed(x)

        # Normalize to unit length for cosine similarity comparison
        return F.normalize(x, p=2, dim=1)

