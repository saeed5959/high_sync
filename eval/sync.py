import argparse
import numpy as np
import os
import subprocess
import tempfile
import shutil


class SyncEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # TODO: Load SyncNet Model here
        # self.model = load_syncnet(...)
        pass

    def run(self, video_path, audio_path):
        """
        Expects paths to a Video file and an Audio file.
        Returns dictionary with metrics.
        """
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

        # --- MOCK LOGIC: Replace with actual SyncNet ---
        # 1. Load Audio (MFCC) from audio_path
        # 2. Load Video (Mouth ROI) from video_path

        # Simulating metrics
        dists = np.random.uniform(5, 12, size=31)  # 31 frames search
        min_idx = np.argmin(dists)
        lse_d = float(dists[min_idx])
        lse_c = float(np.median(dists) - lse_d)
        offset = int(min_idx - 15)
        # --- END MOCK ---

        return {"lse_d": lse_d, "lse_c": lse_c, "offset": offset}


# --- Standalone CLI Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SyncNet Evaluation")
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--audio", default=None, help="Path to audio (optional, will extract if None)")
    args = parser.parse_args()

    evaluator = SyncEvaluator()

    # Logic to handle missing audio arg in CLI mode
    temp_dir = None
    audio_path = args.audio

    try:
        if not audio_path:
            # Create temp audio
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "extracted.wav")
            print(f"Extracting temporary audio to {audio_path}...")
            subprocess.run([
                "ffmpeg", "-y", "-i", args.video,
                "-vn", "-ac", "1", "-ar", "16000", audio_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Run Eval
        res = evaluator.run(args.video, audio_path)
        print(f"Sync Result: {res}")

    finally:
        # Cleanup if we created a temp file
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)