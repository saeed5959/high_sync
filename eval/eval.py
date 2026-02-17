import os
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# Handle local imports
try:
    from data_structs import GroundTruthItem, InferenceItem
    from sync import SyncEvaluator
    from visual import VisualEvaluator
    from semantic import SemanticEvaluator
except ImportError:
    from .data_structs import GroundTruthItem, InferenceItem
    from .sync import SyncEvaluator
    from .visual import VisualEvaluator
    from .semantic import SemanticEvaluator


# ... (AudioExtractor and load_dataset classes remain unchanged) ...
class AudioExtractor:
    def __init__(self):
        self.root = Path(__file__).parent
        self.cache_dir = self.root / "temp_audio_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def extract(self, video_path):
        video_path_obj = Path(video_path).resolve()
        if not video_path_obj.exists(): return None
        safe_name = f"{video_path_obj.parent.name}_{video_path_obj.stem}.wav"
        save_path = self.cache_dir / safe_name
        if not save_path.exists():
            cmd = ["ffmpeg", "-y", "-i", str(video_path_obj), "-vn", "-ac", "1", "-ar", "16000", str(save_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0: return None
            except FileNotFoundError:
                exit("[FATAL ERROR] `ffmpeg` not found.")
        return str(save_path)

    def cleanup(self):
        if self.cache_dir.exists(): shutil.rmtree(self.cache_dir)


def load_dataset(extractor):
    root = Path(__file__).parent / "test_data"
    gt_dir = root / "ground_truth"
    inf_root = root / "inference"
    dataset = []
    gt_files = sorted(list(gt_dir.glob("*.mp4")))
    if not gt_files: return []
    print(f"[Loader] Found {len(gt_files)} Ground Truth identities.")
    for gt_path in gt_files:
        gt_id = gt_path.stem
        transcript = gt_path.with_suffix(".txt").read_text().strip() if gt_path.with_suffix(".txt").exists() else ""
        gt_item = GroundTruthItem(id=gt_id, video_path=str(gt_path), transcript=transcript)
        if inf_root.exists():
            for method_dir in inf_root.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    for inf_file in method_dir.glob("*.mp4"):
                        inf_name = inf_file.stem
                        if (inf_name == gt_id) or (inf_name.startswith(f"{gt_id}_")):
                            inf_audio = extractor.extract(inf_file)
                            if inf_audio is None: continue
                            gt_item.inferences.append(
                                InferenceItem(id=inf_name, video_path=str(inf_file), audio_path=inf_audio,
                                              method_name=method_name))
        dataset.append(gt_item)
    return dataset


# ==========================================
# Main Logic (Updated)
# ==========================================
def main():
    extractor = AudioExtractor()

    try:
        # 1. Load Data
        dataset = load_dataset(extractor)
        if not dataset: return

        # 2. Init Models
        print("Initializing Evaluators...")
        sync_eval = SyncEvaluator()
        vis_eval = VisualEvaluator()
        sem_eval = SemanticEvaluator()

        results = []

        # 3. Process All Videos
        print("Starting Evaluation Loop...")
        for gt in dataset:
            if not gt.inferences:
                continue

            print(f"Processing GT: {gt.id} ({len(gt.inferences)} matches)")

            for inf in gt.inferences:
                # A. SyncNet
                s_res = sync_eval.run(inf.video_path, inf.audio_path)

                # B. Visual Quality (now includes Jerk)
                v_res = vis_eval.run(gt.video_path, inf.video_path)

                # C. Semantic (if transcript exists)
                sem_res = {"wer": 0.0}
                if gt.transcript:
                    sem_res = sem_eval.run(gt.transcript, inf.video_path)

                # Append the results, including the new 'Jerk' metric
                results.append({
                    "Method": inf.method_name,
                    "GT_ID": gt.id,
                    "Inf_ID": inf.id,
                    "LSE-D": s_res['lse_d'],
                    "LSE-C": s_res['lse_c'],
                    "Offset": s_res['offset'],
                    "LDE": v_res['lde'],
                    "FID": v_res['fid'],
                    "Jerk": v_res['jerk'],  # <-- ADDED METRIC
                    "WER": sem_res['wer']
                })

        # 4. Save and Report
        if results:
            df = pd.DataFrame(results)
            print("\n=== Method Averages ===")
            # The new 'Jerk' column will be automatically included in the mean calculation
            print(df.groupby("Method").mean(numeric_only=True))

            df.to_csv("final_results.csv", index=False)
            print("\nSaved detailed results to 'final_results.csv'")
        else:
            print("No matching inferences found.")

    finally:
        # Clean up temp wav files
        extractor.cleanup()


if __name__ == "__main__":
    main()