import os
import cv2
import json
import numpy as np
import mediapipe as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../Datasets/Processed_Data"))
SAVE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../Datasets/Landmarks_Interpolated"))
MAX_FRAMES = 60

IMPORTANT_FACE_IDX = [
    # Eyes
    33, 133, 159, 145, 468, 469,     # Left eye
    263, 362, 386, 374, 471, 472,    # Right eye
    # Eyebrows
    105, 107, 55, 65, 52,             # Left eyebrow
    285, 295, 282, 283, 336,          # Right eyebrow
    # Nose (bridge + tip)
    1, 2, 98, 327, 94, 97, 168, 197,
    # Mouth (outer + inner)
    13, 14, 78, 308, 82, 312,
    87, 317, 88, 95, 178, 191,
    80, 81, 82, 311, 310, 415,
    291, 308, 324, 318, 402, 317
]

def interpolate_sequence(seq, target_len=60):
    T = len(seq)
    if T == 0:
        return np.zeros((target_len, 387))
    if T == target_len:
        return np.array(seq)
    
    seq = np.array(seq)
    D = seq.shape[1]
    
    orig_grid = np.linspace(0, 1, T)
    target_grid = np.linspace(0, 1, target_len)
    
    new_seq = np.zeros((target_len, D))
    for d in range(D):
        new_seq[:, d] = np.interp(target_grid, orig_grid, seq[:, d])
        
    return new_seq

def extract_video_landmarks(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame)

        lm = []

        # Face
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            for idx in IMPORTANT_FACE_IDX:
                p = face[idx]
                lm += [p.x, p.y, p.z]
        else:
            lm += [0] * len(IMPORTANT_FACE_IDX) * 3

        # Left Hand
        if results.left_hand_landmarks:
            for p in results.left_hand_landmarks.landmark:
                lm += [p.x, p.y, p.z]
        else:
            lm += [0] * 21 * 3

        # Right Hand
        if results.right_hand_landmarks:
            for p in results.right_hand_landmarks.landmark:
                lm += [p.x, p.y, p.z]
        else:
            lm += [0] * 21 * 3

        # Pose
        if results.pose_landmarks:
            for p in results.pose_landmarks.landmark:
                lm += [p.x, p.y, p.z]
        else:
            lm += [0] * 33 * 3

        seq.append(lm)

    cap.release()
    return interpolate_sequence(seq, MAX_FRAMES)

def process_word_folder(args):
    """
    Worker function to process all remaining videos in a specific word folder.
    Reuses the MediaPipe Holistic model instance across all videos in the folder.
    """
    view, word_folder, files_to_process_names = args
    input_dir = os.path.normpath(os.path.join(DATA_DIR, view, word_folder))
    output_dir = os.path.normpath(os.path.join(SAVE_DIR, view, word_folder))
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = []
    for f in files_to_process_names:
        save_path = os.path.normpath(os.path.join(output_dir, f.replace(".mp4", ".npy")))
        files_to_process.append((f, save_path))

    print(f"[{view}/{word_folder}] Started processing {len(files_to_process)} videos...", flush=True)

    # Instantiate MediaPipe Holistic ONCE for this worker process
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        model_complexity=1,
        refine_face_landmarks=True # Speedup: disable face mesh refinement (we only need standard 468 mesh indices)
    ) as holistic:
        for idx, (f, save_path) in enumerate(files_to_process, 1):
            video_path = os.path.normpath(os.path.join(input_dir, f))
            try:
                arr = extract_video_landmarks(video_path, holistic)
                np.save(save_path, arr)
            except Exception as e:
                print(f"Error processing {video_path}: {e}", flush=True)
            
            if idx % 10 == 0 or idx == len(files_to_process):
                print(f"[{view}/{word_folder}] Progress: {idx}/{len(files_to_process)} videos done.", flush=True)
                
    return len(files_to_process)

def main():
    views = ["Lateral"]
    all_folders = []
    
    for view in views:
        view_path = os.path.join(DATA_DIR, view)
        if not os.path.exists(view_path):
            continue
        word_folders = os.listdir(view_path)
        for word in word_folders:
            all_folders.append((view, word))
            
    print(f"Found {len(all_folders)} word folders in source dataset.")
    
    # Pre-filter folders and files that are already successfully extracted
    tasks_to_run = []
    skipped_folders = 0
    skipped_files = 0
    
    for view, word in all_folders:
        input_dir = os.path.join(DATA_DIR, view, word)
        output_dir = os.path.join(SAVE_DIR, view, word)
        
        if not os.path.exists(input_dir):
            continue
            
        files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
        
        # Check which files do not have the corresponding .npy file yet
        files_to_process_names = []
        for f in files:
            save_path = os.path.join(output_dir, f.replace(".mp4", ".npy"))
            if not os.path.exists(save_path):
                files_to_process_names.append(f)
                
        if len(files_to_process_names) > 0:
            tasks_to_run.append((view, word, files_to_process_names))
        else:
            skipped_folders += 1
            skipped_files += len(files)
            
    print(f"Skipped {skipped_folders} folders ({skipped_files} files) that are already fully processed.")
    print(f"Remaining folders to process: {len(tasks_to_run)}")
    
    if not tasks_to_run:
        print("All landmarks have been successfully extracted!")
        return
        
    # Limit number of CPU cores to use to prevent out-of-memory crashes (each worker uses ~500MB RAM)
    num_workers = min(10, max(1, os.cpu_count() - 2))
    print(f"Starting parallel extraction with {num_workers} processes...")
    
    total_processed_files = 0
    start_time = time.time()
    total_tasks = len(tasks_to_run)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_word_folder, task): task for task in tasks_to_run}
        
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                processed_count = future.result()
                total_processed_files += processed_count
            except Exception as e:
                print(f"Folder task {task[:2]} generated an exception: {e}")
                processed_count = 0
            
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            eta = (total_tasks - completed) * avg_time
            
            def format_time(seconds):
                mins, secs = divmod(int(seconds), 60)
                hours, mins = divmod(mins, 60)
                if hours > 0:
                    return f"{hours:02d}:{mins:02d}:{secs:02d}"
                return f"{mins:02d}:{secs:02d}"
            
            print(f"[{completed}/{total_tasks}] ({(completed/total_tasks)*100:.1f}%) | "
                  f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)} | "
                  f"Avg: {avg_time:.2f}s/folder | Folder: {task[0]}/{task[1]} ({processed_count} files processed)", flush=True)
                
    print(f"Extraction complete! Processed {total_processed_files} video files in this run.")

if __name__ == "__main__":
    main()
