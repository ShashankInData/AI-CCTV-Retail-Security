"""
Analyze FP and FN clips from the test set.
Saves a contact sheet (grid of frames) for each error clip so you can
visually inspect what confused the model.

Usage:
    python scripts/analyze_errors.py --threshold 0.8
    python scripts/analyze_errors.py --threshold 0.8 --clips-txt data/processed/frame_classification/val/clips.txt
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import VideoMAEForVideoClassification
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_model(model_path, device):
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model


def predict(model, frames_tensor, device):
    """frames_tensor: (1, 16, 3, 224, 224)"""
    with torch.no_grad():
        out = model(pixel_values=frames_tensor.to(device))
        probs = torch.softmax(out.logits, dim=1)[0]
    return probs[1].item()  # shoplifting probability


def make_contact_sheet(frame_paths, label, shoplifting_prob, threshold, out_path, clip_length=16):
    """Create a grid of all 16 frames with label/prob annotation."""
    cols = 4
    rows = clip_length // cols   # 4 rows of 4
    thumb_w, thumb_h = 224, 168
    pad = 4
    header_h = 60

    canvas_w = cols * (thumb_w + pad) + pad
    canvas_h = header_h + rows * (thumb_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    # Header text
    draw = ImageDraw.Draw(canvas)
    truth = "SHOPLIFTING" if label == 1 else "NORMAL"
    pred_label = "SHOPLIFTING" if shoplifting_prob >= threshold else "NORMAL"
    error_type = ""
    if label == 1 and shoplifting_prob < threshold:
        error_type = "FALSE NEGATIVE"
        hdr_color = (255, 80, 80)
    elif label == 0 and shoplifting_prob >= threshold:
        error_type = "FALSE POSITIVE"
        hdr_color = (255, 165, 0)
    else:
        error_type = "CORRECT"
        hdr_color = (80, 200, 80)

    draw.text((pad, 4),  f"{error_type}  |  Truth: {truth}  |  Pred: {pred_label}  ({shoplifting_prob*100:.1f}%)",
              fill=hdr_color)
    src = str(Path(frame_paths[0]).parent.name)
    draw.text((pad, 28), f"Source: {src}", fill=(180, 180, 180))

    # Thumbnails
    for i, p in enumerate(frame_paths[:clip_length]):
        row, col = divmod(i, cols)
        x = pad + col * (thumb_w + pad)
        y = header_h + pad + row * (thumb_h + pad)
        try:
            img = Image.open(p).convert("RGB").resize((thumb_w, thumb_h))
        except Exception:
            img = Image.new("RGB", (thumb_w, thumb_h), (60, 60, 60))
        canvas.paste(img, (x, y))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize FP/FN errors")
    parser.add_argument("--model", default="models/videomae-shoplifting-best")
    parser.add_argument("--clips-txt", default="data/processed/frame_classification/test/clips.txt")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out-dir", default="outputs/error_analysis")
    parser.add_argument("--max-errors", type=int, default=20,
                        help="Max FP+FN contact sheets to save (default: 20)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[Model] Loading {args.model}...")
    model = load_model(args.model, device)
    print("   Done.")

    # Load clips
    clips = []
    with open(args.clips_txt) as f:
        for line in f:
            parts = line.strip().split("\t")
            label = int(parts[0])
            paths = parts[1].split(",")
            clips.append((label, paths))

    out_dir = Path(args.out_dir)
    fp_dir = out_dir / "false_positives"
    fn_dir = out_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    fp_list, fn_list = [], []
    tp = fp = tn = fn_count = 0
    all_probs = []

    print(f"\n[Evaluating] {len(clips)} clips at threshold={args.threshold}")
    for label, paths in tqdm(clips, desc="Running inference"):
        frames = []
        for p in paths[:16]:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(transform(img))
            except Exception:
                break
        if len(frames) < 16:
            continue

        frames_tensor = torch.stack(frames).unsqueeze(0)
        prob = predict(model, frames_tensor, device)
        all_probs.append(prob)
        positive = prob >= args.threshold

        if label == 1 and positive:
            tp += 1
        elif label == 1 and not positive:
            fn_count += 1
            fn_list.append((paths, prob))
        elif label == 0 and positive:
            fp += 1
            fp_list.append((paths, prob))
        else:
            tn += 1

    total = tp + fp + tn + fn_count
    shop = tp + fn_count
    norm = fp + tn
    dr = tp / shop if shop else 0
    fpr = fp / norm if norm else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0

    print(f"\n{'='*50}")
    print(f"  FULL TEST SET  (threshold={args.threshold})")
    print(f"{'='*50}")
    print(f"  Total:            {total}")
    print(f"  Detection Rate:   {dr*100:.1f}%")
    print(f"  FP Rate:          {fpr*100:.1f}%")
    print(f"  Precision:        {prec*100:.1f}%")
    print(f"  F1:               {f1*100:.1f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn_count}")
    print(f"{'='*50}")

    # Save contact sheets
    print(f"\n[Saving error contact sheets -> {out_dir}]")

    print(f"\n  FALSE NEGATIVES ({len(fn_list)}) - missed shoplifting:")
    for i, (paths, prob) in enumerate(fn_list[:args.max_errors]):
        src = Path(paths[0]).parent.name
        print(f"    FN{i+1}: {src}  (prob={prob*100:.1f}%)")
        sheet_path = fn_dir / f"fn_{i+1:02d}_{src}.jpg"
        make_contact_sheet(paths, label=1, shoplifting_prob=prob,
                           threshold=args.threshold, out_path=sheet_path)

    print(f"\n  FALSE POSITIVES ({len(fp_list)}) - normal misclassified as shoplifting:")
    for i, (paths, prob) in enumerate(fp_list[:args.max_errors]):
        src = Path(paths[0]).parent.name
        print(f"    FP{i+1}: {src}  (prob={prob*100:.1f}%)")
        sheet_path = fp_dir / f"fp_{i+1:02d}_{src}.jpg"
        make_contact_sheet(paths, label=0, shoplifting_prob=prob,
                           threshold=args.threshold, out_path=sheet_path)

    print(f"\n[Done] Contact sheets saved to {out_dir}/")
    print(f"       false_positives/  ({len(fp_list)} images)")
    print(f"       false_negatives/  ({len(fn_list)} images)")

    # Probability distribution summary
    all_probs = np.array(all_probs)
    shop_probs = all_probs[[label for label, _ in clips[:len(all_probs)]] == 1] if len(all_probs) > 0 else []
    print(f"\n[Prob distribution]")
    print(f"  All clips   mean={all_probs.mean():.3f}  std={all_probs.std():.3f}")
    labels_arr = np.array([label for label, _ in clips[:len(all_probs)]])
    if labels_arr.sum() > 0:
        print(f"  Shoplifting mean={all_probs[labels_arr==1].mean():.3f}  std={all_probs[labels_arr==1].std():.3f}")
        print(f"  Normal      mean={all_probs[labels_arr==0].mean():.3f}  std={all_probs[labels_arr==0].std():.3f}")


if __name__ == "__main__":
    main()
