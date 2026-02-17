import pandas as pd
from typing import List
from .data_structs import GroundTruthVideo, InferenceVideo
from .sync import SyncEvaluator
from .visual import VisualEvaluator
from .semantic import SemanticEvaluator


class EvalRunner:
    def __init__(self):
        self.sync = SyncEvaluator()
        self.vis = VisualEvaluator()
        self.sem = SemanticEvaluator()

    def evaluate_dataset(self, dataset: List[GroundTruthVideo]) -> pd.DataFrame:
        """
        Loops through all GTs and Inferences, computes metrics, returns DataFrame.
        """
        summary_list = []

        print(f"Starting evaluation on {len(dataset)} GT items...")

        for gt in dataset:
            for inf in gt.inferences:
                print(f"Evaluating {inf.id} ({inf.method_name})...")

                # 1. Sync
                lse_d, lse_c, offset = self.sync.evaluate(inf.video_path, inf.audio_path)

                # 2. Visual (Needs GT video)
                lde, jerk = self.vis.evaluate_landmarks(gt.video_path, inf.video_path)
                fid = self.vis.evaluate_fid(gt.video_path, inf.video_path)

                # 3. Semantic (Needs GT text)
                wer, phoneme = self.sem.evaluate(gt.transcript, inf.video_path)

                # Store in object
                inf.metrics.lse_d = lse_d
                inf.metrics.lse_c = lse_c
                inf.metrics.offset = offset
                inf.metrics.lde = lde
                inf.metrics.jerkiness = jerk
                inf.metrics.fid = fid
                inf.metrics.wer = wer
                inf.metrics.phoneme_acc = phoneme

                # Append to summary for table
                summary_list.append({
                    "Method": inf.method_name,
                    "VideoID": inf.id,
                    "LSE-D": lse_d,
                    "LSE-C": lse_c,
                    "Offset": offset,
                    "LDE": lde,
                    "FID": fid,
                    "WER": wer
                })

        return pd.DataFrame(summary_list)