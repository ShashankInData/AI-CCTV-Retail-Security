from pathlib import Path
import shutil
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

def parse_filename(filename):
    """EXTRACT VIDEO_ID AND FRAME_NUMBER FROM FILENAME"""
    #PATTERN: shoplifting055...
    match = re.match(r'(.+?)_x264_(\d+)\.png', filename)
    if match:
        video_id = match.group(1)
        frame_num = int(match.group(2))
        return video_id, frame_num

    return None, None

def group_frames_into_clips(frame_dir, clip_length = 16, stride =  8):
    """
    Group frames into clips 

    Args:
      frame_dir: Path to directory with frames
      clip_length: Number of frames per clip(16 for VideoMAE)
      stride: Frame skip between clips(8 = 50% overlap)
    Returns:
      List of clips:[(Video_id, [frame_paths]), ...]

      """
    #group frames by video
    videos = defaultdict(list)

    for frames_path in frame_dir.glob("*.png"):
        video_id, frame_num = parse_filename(frames_path.name)
        if video_id:
            videos[video_id].append((frame_num, frames_path))

    ##sort frames with in each video
    for video_id in videos:
        videos[video_id].sort(key=lambda x: x[0])

    ##creat clips
    clips = []
    for video_id, frames in videos.items():
        frame_paths= [f[1] for f in frames]

        ##sliding window to create clips
        for i in range(0,len(frame_paths) - clip_length + 1, stride):
            clip_frames = frame_paths[i:i + clip_length]
            if len(clip_frames) == clip_length:
                clips.append((video_id, clip_frames))

    return clips

def prepare_dataset():
    raw_dir = Path("data/raw/ucf-crime/train")
    output_dir = Path("data/processed/frame_classification")

    print(" scanning shoplifting frames,...")
    shoplifting_clip = group_frames_into_clips(
        raw_dir / "Shoplifting",
        clip_length=16,
        stride=8
    )
    print("Scanning Normal frames..,,")
    normal_clips = group_frames_into_clips(
        raw_dir/ "NormalVideos",
        clip_length=16,
        stride=16
    )

    print(f"\n Clip statistic:")
    print(f"Shoplifting clips: {len(shoplifting_clip)}")
    print(f"Normal clips: {len(normal_clips)}")

    ##balance dataset - undersample normal
    if len(normal_clips) > len(shoplifting_clip) * 1.2:
        normal_sample = random.sample(normal_clips, int(len(shoplifting_clip) * 1.2))
        print(f"Undersampled normal to: {len(normal_sample)} clips")
    else:
        normal_sample = normal_clips
    ##label clips
    all_clips = [(clip, 1) for clip in shoplifting_clip] +\
        [(clip, 0) for clip in normal_sample]    
    random.shuffle(all_clips)

    ##split
    labels = [c[1] for c in all_clips]
    train, temp = train_test_split(all_clips, test_size=0.3, stratify=labels, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=[c[1] for c in temp],random_state=42)

    print(f"\nsplit size:")
    print(f"Train: {len(train)} clips")
    print(f"Val: {len(val)} clips")
    print(f"Test: {len(test)} clips")

    ##save clip metadata 
    ##save lists of frame path 
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "clips.txt", "w") as f:
            for (video_id, frame_path), label in split_data:
                path_str = ",".join([str(p.absolute()) for p in frame_path])
                f.write(f"{label}\t{path_str}\n")
            
    print(f"\n Dataset prepared!")
    print(f"meta data saved to: {output_dir.absolute()}")

    ##print_samples
    print(f"\n sample from train set:")
    with open(output_dir / "train" / "clips.txt", "r") as f:
        sample = f.readline().strip().split("\t")
        label = "shoplifting" if sample[0] == "1" else "Normal"
        num_frames = len(sample[1].split(","))
        print(f"label: {label}, Frames: {num_frames}")

if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()    