import argparse
import os
import shutil
import subprocess
import tempfile
from typing import Tuple

import cv2
import librosa
import numpy as np


class SyncEvaluator:
    """
    Lightweight lip-sync evaluator based on audio/video motion correlation.

    Unlike the placeholder version, this implementation performs an actual
    signal analysis pipeline:

    1. Extracts a rough mouth-region motion signal from the video.
    2. Computes frame-aligned audio energy.
    3. Uses normalized cross-correlation to estimate sync confidence.

    Metrics returned (all floats except offset):
        - lse_d : 5 (perfect) → 15+ (bad). Derived from corr score.
        - lse_c : correlation confidence in [0, 1].
        - offset: Estimated frame offset (audio lag, positive if audio is late).

    NOTE: This is a deterministic heuristic. For research-grade SyncNet
    evaluation, replace the `_estimate_sync()` implementation with a
    pretrained SyncNet model.
    """

    def __init__(self, mouth_crop_ratio: float = 0.33):
        """
        Args:
            mouth_crop_ratio: Fraction of frame height considered as mouth region
                              (bottom portion). Default = lower third.
        """
        self.mouth_crop_ratio = mouth_crop_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, video_path: str, audio_path: str) -> dict:
        """
        Args:
            video_path: Path to the video file.
            audio_path: Path to the audio file (mono WAV preferred).

        Returns:
            dict with keys: lse_d, lse_c, offset.
        """
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

        video_signal, fps = self._extract_video_motion(video_path)
        audio_signal = self._extract_audio_energy(audio_path, fps)

        if len(video_signal) < 10 or len(audio_signal) < 10:
            return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

        lse_c, lse_d, offset = self._estimate_sync(video_signal, audio_signal)

        return {"lse_d": lse_d, "lse_c": lse_c, "offset": offset}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _extract_video_motion(self, video_path: str) -> Tuple[np.ndarray, float]:
        """Return per-frame mouth-motion energy and FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), 25.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return np.array([]), fps

        prev_roi = self._mouth_roi(prev_frame)
        motion_series = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            curr_roi = self._mouth_roi(frame)
            diff = cv2.absdiff(curr_roi, prev_roi)
            motion = float(np.mean(diff))
            motion_series.append(motion)
            prev_roi = curr_roi

        cap.release()
        motion_series = np.array(motion_series, dtype=np.float32)
        if motion_series.size:
            motion_series = self._normalize_signal(motion_series)
        return motion_series, fps

    def _mouth_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop bottom portion of frame and convert to grayscale."""
        h, w = frame.shape[:2]
        mouth_top = int(max(0, h * (1 - self.mouth_crop_ratio)))
        roi = frame[mouth_top:h, w // 4 : 3 * w // 4]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return gray

    def _extract_audio_energy(self, audio_path: str, fps: float) -> np.ndarray:
        """Compute frame-aligned audio RMS energy."""
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        if wav.size == 0:
            return np.array([])

        samples_per_frame = max(1, int(sr / max(fps, 1e-6)))
        energies = []
        total_frames = int(len(wav) / samples_per_frame)

        for i in range(total_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            chunk = wav[start:end]
            if chunk.size == 0:
                break
            rms = np.sqrt(np.mean(chunk**2) + 1e-12)
            energies.append(rms)

        energies = np.array(energies, dtype=np.float32)
        if energies.size:
            energies = self._normalize_signal(energies)
        return energies

    def _estimate_sync(self, video_sig: np.ndarray, audio_sig: np.ndarray):
        """Compute normalized cross-correlation metrics."""
        length = min(len(video_sig), len(audio_sig))
        if length == 0:
            return 0.0, 99.0, 0

        v = video_sig[:length]
        a = audio_sig[:length]

        corr = np.correlate(v, a, mode="full")
        corr /= (length + 1e-6)

        max_idx = int(np.argmax(corr))
        max_corr = float(np.clip(corr[max_idx], 0.0, 1.0))

        offset_frames = max_idx - (length - 1)

        # Map correlation to SyncNet-like range (≈5 good, ≈15 bad)
        lse_d = float((1.0 - max_corr) * 10.0 + 5.0)
        lse_c = max_corr

        return lse_c, lse_d, offset_frames

    @staticmethod
    def _normalize_signal(sig: np.ndarray) -> np.ndarray:
        sig = sig - sig.mean()
        denom = sig.std() + 1e-6
        return sig / denom


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Heuristic SyncNet-style Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file. If omitted, mono audio is extracted from the video.",
    )
    args = parser.parse_args()

    evaluator = SyncEvaluator()
    temp_dir = None
    audio_path = args.audio

    try:
        if not audio_path:
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "extracted.wav")
            print(f"[SyncEvaluator] Extracting audio to {audio_path}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    args.video,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    audio_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        result = evaluator.run(args.video, audio_path)
        print(f"Sync Result: {result}")

    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()