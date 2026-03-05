import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torchvision import transforms


def get_augmentation_transforms(num_channels=3):
    """
    Get standard data augmentation transforms for image classification.
    
    Args:
        num_channels: Number of image channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Augmentation transform composition
    """
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR)
])

class FitsDataset(Dataset):
    """
    Dataset for FITS galaxy images with multilabel ring classification.
    
    Labels are converted from ring_class (0-3) to multilabel format:
    - 0 → [0, 0]: No rings
    - 1 → [1, 0]: Inner ring only
    - 2 → [0, 1]: Outer ring only
    - 3 → [1, 1]: Both rings (inner + outer)
    """
    
    # Multilabel mapping: ring_class → [inner_ring, outer_ring]
    RING_CLASS_TO_MULTILABEL = {
        0: [0, 0],  # No rings
        1: [1, 0],  # Inner ring only
        2: [0, 1],  # Outer ring only
        3: [1, 1]   # Both rings
    }
    
    def __init__(self, file_paths, labels, transform=None, augmentation_transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augmentation_transform = augmentation_transform

    def _preprocess_fits_bands(self, data):
        # Handle NaNs and apply arcsinh scaling
        data = np.nan_to_num(data)
        # Using a standard scaling factor; adjust based on your data intensity
        data = np.arcsinh(data) 
        
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        return data.astype(np.float32)
    # def _preprocess_fits_bands(self, data):
    #     """Preprocess individual bands before Lupton RGB conversion."""
    #     # Handle NaNs
    #     data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    #     # Clip negative values
    #     data = np.clip(data, 0, None)
    #     return data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with fits.open(file_path) as hdul:
            data = hdul[0].data

        data = np.asarray(data, dtype=np.float32)
        image_tensor = torch.from_numpy(data)

        data = self._preprocess_fits_bands(data)
        
        # Apply preprocessing transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Apply augmentation transforms if provided; otherwise, apply standard resizing and normalization transforms
        if self.augmentation_transform:
            image_tensor = self.augmentation_transform(image_tensor)
        else:
            standard_transform = transforms.Compose([transforms.Resize((244, 244)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image_tensor = standard_transform(image_tensor)
        
        # Convert ring_class (0-3) to binary multilabel format [inner_ring, outer_ring]
        ring_class = self.labels[idx]
        binary_label = self.RING_CLASS_TO_MULTILABEL[ring_class]
        label = torch.tensor(binary_label, dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'ring_class': label  # Shape: [2] → [inner_ring, outer_ring]
        }

class ZoobotFitsDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, img_dir=None, batch_size=32, num_workers=4, transform_pipeline = None, use_augmentation=False):
        """
        Args:
            csv_path (str): Path to the catalog CSV.
            img_dir (str, optional): Root directory if file_loc is relative.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker processes.
            transform_pipeline (torchvision.transforms, optional): Preprocessing transforms to apply to all datasets.
            use_augmentation (bool): Whether to apply data augmentation to training set.
        """
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation

        self.transform = transform_pipeline if transform_pipeline else transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.augmentation_transform = get_augmentation_transforms(num_channels=3) if use_augmentation else None

    def setup(self, stage=None):
        """
        Setup datasets with class balancing.
        
        Multilabel Mapping:
        - ring_class 0 → [inner_ring=0, outer_ring=0] → No rings
        - ring_class 1 → [inner_ring=1, outer_ring=0] → Inner ring only
        - ring_class 2 → [inner_ring=0, outer_ring=1] → Outer ring only
        - ring_class 3 → [inner_ring=1, outer_ring=1] → Both rings
        """
        # Multilabel class mapping
        RING_CLASS_TO_MULTILABEL = {
            0: [0, 0],  # No rings
            1: [1, 0],  # Inner ring only
            2: [0, 1],  # Outer ring only
            3: [1, 1]   # Both rings
        }
        
        df = pd.read_csv(self.csv_path)
        
        paths = df['file_loc'].tolist()
        labels = df['ring_class'].values

        # Split: Train 80%, Val 10%, Test 10%
        # Stratify by ring_class to maintain class distribution across splits
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )

        # Balance training set by oversampling to 3000 samples per class (12000 total)
        # This ensures equal representation of all 4 multilabel combinations
        target_samples_per_class = 3000
        balanced_paths = []
        balanced_labels = []
        
        for class_id in range(4):  # Iterate through all 4 multilabel combinations
            multilabel = RING_CLASS_TO_MULTILABEL[class_id]
            
            # Get indices for this class
            class_indices = [i for i, label in enumerate(train_labels) if label == class_id]
            current_count = len(class_indices)
            
            # Oversample with replacement to reach target
            sampled_indices = np.random.choice(
                class_indices, 
                size=target_samples_per_class, 
                replace=True
            )
            
            # Add sampled paths and labels
            for idx in sampled_indices:
                balanced_paths.append(train_paths[idx])
                balanced_labels.append(train_labels[idx])
            
            print(f"Class {class_id} {multilabel}: {current_count} → {target_samples_per_class} samples")
        
        # Replace with balanced versions
        train_paths = balanced_paths
        train_labels = np.array(balanced_labels)

        # Create datasets for ALL stages
        # FitsDataset will convert ring_class to multilabel format [inner_ring, outer_ring]
        self.train_ds = FitsDataset(train_paths, train_labels, self.transform, 
                                     augmentation_transform=self.augmentation_transform)
        self.val_ds = FitsDataset(val_paths, val_labels, self.transform)
        self.test_ds = FitsDataset(test_paths, test_labels, self.transform)
        self.predict_ds = FitsDataset(test_paths, test_labels, self.transform)



    def train_dataloader(self):
        # Dataset is already balanced in setup(), use regular shuffling
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, num_workers=self.num_workers)