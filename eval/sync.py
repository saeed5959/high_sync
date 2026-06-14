import argparse
import logging
import os
import subprocess
import tempfile
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile


LOGGER = logging.getLogger("highsync.eval.sync")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


class SyncEvaluator:
    """
    Lightweight lip-sync evaluator based on audio/video motion correlation.

    If audio_path is None during run(), this class automatically extracts
    audio from the video file using ffmpeg.
    """

    def __init__(self, mouth_crop_ratio: float = 0.33):
        self.mouth_crop_ratio = mouth_crop_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, video_path: str, audio_path: Optional[str] = None) -> dict:
        """
        Args:
            video_path: Path to the video file.
            audio_path: Path to the audio file. If None, audio is extracted
                        from video_path to a temp file.

        Returns:
            dict with keys: lse_d, lse_c, offset.
        """
        run_start = time.perf_counter()
        if not os.path.exists(video_path):
            print(f"[SyncEvaluator] Video not found: {video_path}")
            LOGGER.info("Timing | sync_run missing_video total=%s", format_duration(time.perf_counter() - run_start))
            return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

        temp_audio_file = None
        audio_extract_seconds = 0.0
        video_motion_seconds = 0.0
        audio_energy_seconds = 0.0
        estimate_seconds = 0.0

        try:
            # --- 1. Audio Handling Logic ---
            if audio_path is None:
                # Create a temporary file for audio extraction
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)  # Close file descriptor so ffmpeg can write to it
                temp_audio_file = temp_path
                audio_path = temp_path

                # Extract audio using ffmpeg
                section_start = time.perf_counter()
                self._extract_audio_from_video(video_path, temp_audio_file)
                audio_extract_seconds = time.perf_counter() - section_start
            elif not os.path.exists(audio_path):
                print(f"[SyncEvaluator] Audio file not found: {audio_path}")
                LOGGER.info("Timing | sync_run missing_audio total=%s", format_duration(time.perf_counter() - run_start))
                return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

            # --- 2. Signal Extraction ---
            section_start = time.perf_counter()
            video_signal, fps = self._extract_video_motion(video_path)
            video_motion_seconds = time.perf_counter() - section_start
            section_start = time.perf_counter()
            audio_signal = self._extract_audio_energy(audio_path, fps)
            audio_energy_seconds = time.perf_counter() - section_start

            if len(video_signal) < 10 or len(audio_signal) < 10:
                LOGGER.info(
                    "Timing | sync_run video=%s audio_extract=%s video_motion=%s audio_energy=%s estimate=%s total=%s note=short_signal video_len=%d audio_len=%d",
                    os.path.basename(video_path),
                    format_duration(audio_extract_seconds),
                    format_duration(video_motion_seconds),
                    format_duration(audio_energy_seconds),
                    format_duration(estimate_seconds),
                    format_duration(time.perf_counter() - run_start),
                    len(video_signal),
                    len(audio_signal),
                )
                return {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}

            # --- 3. Compute Metrics ---
            section_start = time.perf_counter()
            lse_c, lse_d, offset = self._estimate_sync(video_signal, audio_signal)
            estimate_seconds = time.perf_counter() - section_start

            LOGGER.info(
                "Timing | sync_run video=%s audio_extract=%s video_motion=%s audio_energy=%s estimate=%s total=%s video_len=%d audio_len=%d fps=%.2f",
                os.path.basename(video_path),
                format_duration(audio_extract_seconds),
                format_duration(video_motion_seconds),
                format_duration(audio_energy_seconds),
                format_duration(estimate_seconds),
                format_duration(time.perf_counter() - run_start),
                len(video_signal),
                len(audio_signal),
                fps,
            )

            return {"lse_d": lse_d, "lse_c": lse_c, "offset": offset}

        finally:
            # --- 4. Cleanup ---
            if temp_audio_file and os.path.exists(temp_audio_file):
                cleanup_start = time.perf_counter()
                os.remove(temp_audio_file)
                LOGGER.debug("Timing | sync_temp_cleanup=%s", format_duration(time.perf_counter() - cleanup_start))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _extract_audio_from_video(self, video_path: str, output_path: str):
        """Runs ffmpeg to extract mono 16kHz wav."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}")

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
            # Simple frame differencing as motion proxy
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
        # Simple heuristic: center-bottom crop
        roi = frame[mouth_top:h, w // 4: 3 * w // 4]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return gray

    def _extract_audio_energy(self, audio_path: str, fps: float) -> np.ndarray:
        """Compute frame-aligned audio RMS energy."""
        try:
            wav, sr = sf.read(audio_path, always_2d=False)
        except Exception:
            try:
                sr, wav = wavfile.read(audio_path)
            except Exception:
                return np.array([])

        wav = np.asarray(wav)
        if wav.size == 0:
            return np.array([])
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if np.issubdtype(wav.dtype, np.integer):
            wav /= float(np.iinfo(wav.dtype).max)
        wav = wav.astype(np.float32)
        if sr != 16000:
            target_len = int(round(len(wav) * 16000 / sr))
            wav = signal.resample(wav, target_len).astype(np.float32)
            sr = 16000

        samples_per_frame = max(1, int(sr / max(fps, 1e-6)))
        energies = []
        total_frames = int(len(wav) / samples_per_frame)

        for i in range(total_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            chunk = wav[start:end]
            if chunk.size == 0:
                break
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
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

        # Offset: 0 is center. positive = audio leads? check alignment logic.
        # Usually argmax - (len-1).
        offset_frames = max_idx - (length - 1)

        # Map correlation to SyncNet-like range (≈5 good, ≈15 bad)
        # Heuristic: High correlation (1.0) -> Low distance (5.0)
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
        help="Path to audio file. If omitted, extracted automatically.",
    )
    args = parser.parse_args()

    evaluator = SyncEvaluator()

    # The class now handles the extraction logic safely
    result = evaluator.run(args.video, args.audio)

    print(f"Sync Result: {result}")


if __name__ == "__main__":
    main()
