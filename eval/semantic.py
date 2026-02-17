import argparse
import os


class SemanticEvaluator:
    def __init__(self):
        # TODO: Load ASR/LipReading Model
        pass

    def run(self, transcript_text, video_path):
        """
        Returns WER.
        """
        if not transcript_text:
            return {"wer": 0.0}

        # Mock Logic
        # pred_text = lip_read(video_path)
        pred_text = "hello world"

        # dist = editdistance.eval(...)
        dist = 1
        n = len(transcript_text.split())
        wer = dist / n if n > 0 else 1.0

        return {"wer": wer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    evaluator = SemanticEvaluator()
    res = evaluator.run(args.trans, args.video)
    print(f"Semantic Result: {res}")