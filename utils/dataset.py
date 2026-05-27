"""
PyTorch Dataset for brain age prediction with proper normalization and augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import torchvision.transforms.functional as TF
import random

class BrainAgeDataset(Dataset):
    def __init__(self, data_dir='data/processed', augment=False, use_3_slices=True):
        """
        Brain Age Dataset with age normalization
        
        Args:
            data_dir: Directory with processed .npz files
            augment: Apply data augmentation (training only)
            use_3_slices: Use 3 consecutive slices (True) or single slice (False)
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.use_3_slices = use_3_slices
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Calculate age statistics for normalization
        ages = [m['age'] for m in self.metadata]
        self.age_mean = np.mean(ages)
        self.age_std = np.std(ages)
        self.age_min = min(ages)
        self.age_max = max(ages)
        
        print(f"📚 Dataset: {len(self.metadata)} subjects")
        print(f"   Age range: {self.age_min:.1f}-{self.age_max:.1f} years")
        print(f"   Age mean±std: {self.age_mean:.1f}±{self.age_std:.1f} years")
        print(f"   Augmentation: {'ON' if augment else 'OFF'}")
        print(f"   Slices: {'3-channel' if use_3_slices else '1-channel'}")
    
    def __len__(self):
        return len(self.metadata)
    
    def normalize_age(self, age):
        """Normalize age to ~N(0,1) distribution"""
        return (age - self.age_mean) / self.age_std
    
    def denormalize_age(self, age_normalized):
        """Convert normalized age back to years"""
        return age_normalized * self.age_std + self.age_mean
    
    def __getitem__(self, idx):
        # Load subject data
        subject = self.metadata[idx]
        data = np.load(self.data_dir / f"{subject['subject_id']}.npz")
        
        slices = data['slices'].astype(np.float32)  # (40, 224, 224)
        age = float(data['age'])
        
        # Normalize age (CRITICAL for training stability)
        age_normalized = self.normalize_age(age)
        
        # Select slices
        if self.use_3_slices:
            # Use 3 consecutive middle slices
            middle_idx = 20
            
            # Random slice selection during training
            if self.augment:
                shift = random.randint(-5, 5)
                middle_idx = np.clip(middle_idx + shift, 1, 38)
            
            brain_img = np.stack([
                slices[middle_idx - 1],
                slices[middle_idx],
                slices[middle_idx + 1]
            ], axis=0)  # Shape: (3, 224, 224)
        else:
            # Single middle slice
            middle_idx = 20
            if self.augment:
                shift = random.randint(-5, 5)
                middle_idx = np.clip(middle_idx + shift, 0, 39)
            brain_img = slices[middle_idx][np.newaxis, ...]  # (1, 224, 224)
        
        # Convert to tensor
        brain_img = torch.from_numpy(brain_img).float()
        
        # Apply augmentation
        if self.augment:
            brain_img = self.apply_augmentation(brain_img)
        
        age_tensor = torch.tensor(age_normalized, dtype=torch.float32)
        
        return brain_img, age_tensor, subject['subject_id']
    
    def apply_augmentation(self, image):
        """
        Apply random augmentations to brain MRI
        
        Args:
            image: Tensor (C, H, W)
        Returns:
            Augmented tensor
        """
        # 1. Random horizontal flip (50%)
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
        
        # 2. Random rotation (-10° to +10°)
        if random.random() > 0.3:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # 3. Random translation and scaling
        if random.random() > 0.5:
            translate = [random.randint(-15, 15), random.randint(-15, 15)]
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(
                image, 
                angle=0, 
                translate=translate, 
                scale=scale, 
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR
            )
        
        # 4. Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.015
            image = image + noise
        
        # 5. Brightness/contrast adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.95, 1.05)
            image = image * brightness
            
            contrast = random.uniform(0.95, 1.05)
            mean = image.mean()
            image = (image - mean) * contrast + mean
        
        return image
    
    def get_age_stats(self):
        """Return age statistics for denormalization"""
        return {
            'mean': self.age_mean,
            'std': self.age_std,
            'min': self.age_min,
            'max': self.age_max
        }