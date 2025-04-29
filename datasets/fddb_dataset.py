import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FDDBFaceDataset(Dataset):
    """
    Custom Dataset for loading cropped FDDB face images for classification.

    Args:
        csv_file (str): Path to CSV file containing image filenames and labels.
        img_dir (str): Directory where cropped face images are stored.
        transform (callable, optional): Transformations to apply to each image.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        # Load annotations (filenames + labels) from CSV
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # Remap labels to a continuous range [0, N-1]
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.annotations['label'] = self.annotations['label'].map(self.label_map)

        # Image transformations (resize, tensor conversion, normalization)
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Fetch a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            (Tensor, int): Tuple containing the transformed image and its integer label.
        """
        # Build full path to image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]['filename'])

        # Load image and convert to RGB format
        image = Image.open(img_path).convert("RGB")

        # Retrieve corresponding label (already mapped to integer)
        label = int(self.annotations.iloc[idx]['label'])  # ensure label is int

        # Apply preprocessing transformations
        if self.transform:
            image = self.transform(image)

        return image, label