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
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_test(model, dataloader, criterion, device):
    """Comprehensive evaluation on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=frames)
            loss = criterion(outputs.logits, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(outputs.logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall accuracy
    accuracy = correct / total
    
    # Per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Normal class metrics
    # For normal class (class 0), treating it as positive:
    #   TP_normal = predicted 0 AND actual 0 = TN
    #   FP_normal = predicted 0 AND actual 1 = FN (predicted normal but was shoplifting)
    #   FN_normal = predicted 1 AND actual 0 = FP (predicted shoplifting but was normal)
    # Precision: of all predicted normal, how many were actually normal? = TN / (TN + FP)
    #   Denominator (TN + FP) = all cases where we predicted normal (predicted = 0)
    # Recall: of all actual normal, how many did we correctly identify? = TN / (TN + FN)
    #   Denominator (TN + FN) = all cases where actual was normal (actual = 0)
    precision_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0
    
    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision_shoplifting': precision,
        'recall_shoplifting': recall,
        'f1_shoplifting': f1,
        'precision_normal': precision_normal,
        'recall_normal': recall_normal,
        'f1_normal': f1_normal,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }

def print_results(results):
    """Print formatted results"""
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n[Overall Metrics]")
    print(f"   Test Loss:     {results['loss']:.4f}")
    print(f"   Test Accuracy: {100*results['accuracy']:.2f}%")
    
    print(f"\n[Shoplifting Class (Positive)]")
    print(f"   Precision: {100*results['precision_shoplifting']:.2f}%")
    print(f"   Recall:    {100*results['recall_shoplifting']:.2f}%")
    print(f"   F1-Score:  {100*results['f1_shoplifting']:.2f}%")
    
    print(f"\n[Normal Class (Negative)]")
    print(f"   Precision: {100*results['precision_normal']:.2f}%")
    print(f"   Recall:    {100*results['recall_normal']:.2f}%")
    print(f"   F1-Score:  {100*results['f1_normal']:.2f}%")
    
    print(f"\n[Confusion Matrix]")
    cm = results['confusion_matrix']
    print(f"                  Predicted")
    print(f"                Normal  Shoplifting")
    print(f"   Actual Normal   {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"   Shoplifting     {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    print(f"\n[Classification Report]")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['Normal', 'Shoplifting'],
        digits=4
    ))
    
    # Show some high-confidence predictions
    print(f"\n[Sample High-Confidence Predictions]")
    probs = results['probabilities']
    preds = results['predictions']
    labels = results['labels']
    
    # Find high confidence shoplifting predictions
    shoplifting_confident = np.where((preds == 1) & (probs[:, 1] > 0.95))[0][:5]
    normal_confident = np.where((preds == 0) & (probs[:, 0] > 0.95))[0][:5]
    
    print(f"\n   Top 5 High-Confidence Shoplifting Predictions:")
    for idx in shoplifting_confident:
        correct = "[OK]" if labels[idx] == 1 else "[X]"
        print(f"   {correct} Sample {idx}: Predicted=Shoplifting ({100*probs[idx, 1]:.2f}%), Actual={'Shoplifting' if labels[idx] == 1 else 'Normal'}")
    
    print(f"\n   Top 5 High-Confidence Normal Predictions:")
    for idx in normal_confident:
        correct = "[OK]" if labels[idx] == 0 else "[X]"
        print(f"   {correct} Sample {idx}: Predicted=Normal ({100*probs[idx, 0]:.2f}%), Actual={'Normal' if labels[idx] == 0 else 'Shoplifting'}")

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Device] Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = FrameClipDataset(
        "data/processed/frame_classification/test/clips.txt",
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=4)
    
    print(f"\n[Dataset] Test dataset size: {len(test_dataset)} clips")
    
    # Load trained model
    print(f"\n[Model] Loading trained model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "models/videomae-shoplifting-best"
    )
    model = model.to(DEVICE)
    model.eval()
    
    print(f"   Model loaded successfully!")
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    results = evaluate_test(model, test_loader, criterion, DEVICE)
    
    # Print results
    print_results(results)
    
    print("\n" + "="*70)
    print("[OK] Testing complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
