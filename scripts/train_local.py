import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.clip_dataset import FrameClipDataset
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for frames, labels in pbar:
        # frames: (batch, num_frames, C, H, W) - this is the correct format for VideoMAE
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=frames)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Evaluating"):
            # frames: (batch, num_frames, C, H, W) - this is the correct format for VideoMAE
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=frames)
            loss = criterion(outputs.logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate per-class metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Shoplifting class (1) metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return running_loss / len(dataloader), correct / total, precision, recall, f1

def main():
    # Config
    BATCH_SIZE = 4  # reduced to fit full fine-tune in VRAM
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5  # lower LR for full fine-tuning
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🖥️  Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = FrameClipDataset(
        "data/processed/frame_classification/train/clips.txt",
        transform=transform
    )
    val_dataset = FrameClipDataset(
        "data/processed/frame_classification/val/clips.txt",
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} clips")
    print(f"   Val:   {len(val_dataset)} clips")
    
    # Model
    print(f"\n🤖 Loading VideoMAE model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=2,
        id2label={0: "normal", 1: "shoplifting"},
        label2id={"normal": 0, "shoplifting": 1},
        ignore_mismatched_sizes=True
    )
    
    # Full fine-tune: unfreeze all layers with layer-wise LR decay
    print("🔓 Full fine-tuning all layers...")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params:,}")

    model = model.to(DEVICE)

    # Loss with class weights to handle any remaining imbalance
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(DEVICE))

    # Separate LRs: lower for backbone, higher for classifier head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE},
        {"params": head_params, "lr": LEARNING_RATE * 10},
    ], weight_decay=1e-4)
    
    # Training loop
    best_val_f1 = 0
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {100*val_acc:.2f}%")
        print(f"Val Precision: {100*val_precision:.2f}% | Recall: {100*val_recall:.2f}% | F1: {100*val_f1:.2f}%")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"💾 New best F1! Saving model...")
            model.save_pretrained("models/videomae-shoplifting-best")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, "models/videomae-shoplifting-best/checkpoint.pt")
        
        print()
    
    print(f"\n✅ Training complete!")
    print(f"   Best Val F1: {100*best_val_f1:.2f}%")
    print(f"   Model saved to: models/videomae-shoplifting-best/")

if __name__ == "__main__":
    main()
