import argparse
import logging
import math
import re
import time
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("highsync.eval.semantic")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


class SemanticEvaluator:
    """
    Computes WER from real transcripts.

    The evaluator does not invent ASR output. It uses, in order:
      1. pred_text passed by the caller,
      2. a .txt file next to the generated video,
      3. an explicitly loaded Whisper model.

    If none is available, WER is NaN and the note explains why.
    """

    def __init__(self, whisper_model_name: Optional[str] = None, device: str = "cpu"):
        init_start = time.perf_counter()
        self.whisper_model = None
        self.whisper_note = ""
        self.device = device
        if whisper_model_name:
            try:
                import whisper

                section_start = time.perf_counter()
                self.whisper_model = whisper.load_model(whisper_model_name, device=device)
                LOGGER.info(
                    "Timing | semantic_init whisper_load=%s model=%s",
                    format_duration(time.perf_counter() - section_start),
                    whisper_model_name,
                )
            except Exception as exc:
                self.whisper_note = f"whisper_unavailable:{exc}"
        LOGGER.info("Timing | semantic_init total=%s", format_duration(time.perf_counter() - init_start))

    def run(self, transcript_text: str, video_path: str, pred_text: Optional[str] = None) -> dict:
        run_start = time.perf_counter()
        if not transcript_text:
            LOGGER.info("Timing | semantic_run video=%s total=%s note=missing_reference_transcript", Path(video_path).name, format_duration(time.perf_counter() - run_start))
            return {"wer": math.nan, "pred_text": "", "semantic_note": "missing_reference_transcript"}

        prediction = pred_text
        note = "explicit_prediction_text"
        sidecar_seconds = 0.0
        whisper_seconds = 0.0
        wer_seconds = 0.0

        if prediction is None:
            sidecar = Path(video_path).with_suffix(".txt")
            section_start = time.perf_counter()
            if sidecar.exists():
                prediction = sidecar.read_text(encoding="utf-8").strip()
                note = "prediction_from_sidecar_txt"
            sidecar_seconds = time.perf_counter() - section_start

        if prediction is None and self.whisper_model is not None:
            section_start = time.perf_counter()
            result = self.whisper_model.transcribe(video_path)
            whisper_seconds = time.perf_counter() - section_start
            prediction = str(result.get("text", "")).strip()
            note = "prediction_from_whisper"

        if prediction is None:
            if self.whisper_note:
                note = self.whisper_note
            else:
                note = "missing_prediction_transcript_or_asr_model"
            LOGGER.info(
                "Timing | semantic_run video=%s sidecar=%s whisper=%s wer=%s total=%s note=%s",
                Path(video_path).name,
                format_duration(sidecar_seconds),
                format_duration(whisper_seconds),
                format_duration(wer_seconds),
                format_duration(time.perf_counter() - run_start),
                note,
            )
            return {"wer": math.nan, "pred_text": "", "semantic_note": note}

        section_start = time.perf_counter()
        wer = self._wer(transcript_text, prediction)
        wer_seconds = time.perf_counter() - section_start
        LOGGER.info(
            "Timing | semantic_run video=%s sidecar=%s whisper=%s wer=%s total=%s note=%s",
            Path(video_path).name,
            format_duration(sidecar_seconds),
            format_duration(whisper_seconds),
            format_duration(wer_seconds),
            format_duration(time.perf_counter() - run_start),
            note,
        )
        return {"wer": wer, "pred_text": prediction, "semantic_note": note}

    @staticmethod
    def _normalize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    @classmethod
    def _wer(cls, reference: str, hypothesis: str) -> float:
        ref = cls._normalize(reference)
        hyp = cls._normalize(hypothesis)
        if not ref:
            return math.nan

        dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            dp[i][0] = i
        for j in range(len(hyp) + 1):
            dp[0][j] = j

        for i, ref_word in enumerate(ref, start=1):
            for j, hyp_word in enumerate(hyp, start=1):
                cost = 0 if ref_word == hyp_word else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[-1][-1] / len(ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcript-based semantic evaluation")
    parser.add_argument("--trans", required=True, help="Reference transcript text")
    parser.add_argument("--video", required=True, help="Path to evaluated video")
    parser.add_argument("--pred-text", default=None, help="Predicted transcript text")
    parser.add_argument("--pred-file", default=None, help="Text file containing predicted transcript")
    parser.add_argument("--whisper-model", default=None, help="Optional local Whisper model name/path")
    parser.add_argument("--device", default="cpu", help="Device for Whisper, e.g. cpu or cuda:0")
    args = parser.parse_args()

    pred = args.pred_text
    if pred is None and args.pred_file:
        pred = Path(args.pred_file).read_text(encoding="utf-8").strip()

    evaluator = SemanticEvaluator(whisper_model_name=args.whisper_model, device=args.device)
    res = evaluator.run(args.trans, args.video, pred_text=pred)
    print(f"Semantic Result: {res}")
