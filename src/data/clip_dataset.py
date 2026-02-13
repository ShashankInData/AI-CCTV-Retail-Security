import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np

class FrameClipDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        """
        Args:
            metadata_file: Path to clips.txt
            transform: torchvision transforms for frames
        """
        self.transform = transform
        self.clips = []
        
        # Load metadata
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                label = int(parts[0])
                frame_paths = parts[1].split(",")
                self.clips.append((frame_paths, label))
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        frame_paths, label = self.clips[idx]
        
        # Load frames
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # Stack into tensor: (num_frames, C, H, W)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label

if __name__ == "__main__":
    from torchvision import transforms
    from pathlib import Path
    
    # Test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FrameClipDataset(
        "data/processed/frame_classification/train/clips.txt",
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    frames, label = dataset[0]
    print(f"Frames shape: {frames.shape}")  # Should be [16, 3, 224, 224]
    print(f"Label: {'Shoplifting' if label == 1 else 'Normal'}")
    
    print("\n[OK] Dataset test passed!")
