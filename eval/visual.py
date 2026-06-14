import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


LIP_LANDMARKS = np.array(
    [
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
        95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
        312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    ],
    dtype=np.int32,
)
LOGGER = logging.getLogger("highsync.eval.visual")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


class VisualEvaluator:
    """
    Deterministic visual evaluator for generated talking-head videos.

    LDE is only reported when MediaPipe Face Landmarker is installed and the
    local task model is available. FID is intentionally left as NaN unless a
    real Inception/FVD feature extractor is added; returning random FID would
    make the results unusable for a paper.
    """

    def __init__(self, device: str = "cpu", frame_stride: int = 1, crop_size: int = 128):
        init_start = time.perf_counter()
        self.device = device
        self.frame_stride = max(1, int(frame_stride))
        self.crop_size = int(crop_size)
        section_start = time.perf_counter()
        self.face_detector = self._make_opencv_face_detector()
        LOGGER.info("Timing | visual_init face_detector=%s", format_duration(time.perf_counter() - section_start))
        section_start = time.perf_counter()
        self.landmarker = self._make_mediapipe_landmarker()
        LOGGER.info(
            "Timing | visual_init mediapipe_landmarker=%s available=%s total=%s",
            format_duration(time.perf_counter() - section_start),
            self.landmarker is not None,
            format_duration(time.perf_counter() - init_start),
        )

    def run(self, gt_path: str, inf_path: str) -> dict:
        run_start = time.perf_counter()
        if not os.path.exists(gt_path) or not os.path.exists(inf_path):
            LOGGER.info("Timing | visual_run missing_input total=%s", format_duration(time.perf_counter() - run_start))
            return self._empty_result("missing_input_video")

        gt_lip_crops: List[np.ndarray] = []
        inf_lip_crops: List[np.ndarray] = []
        gt_landmarks: List[np.ndarray] = []
        inf_landmarks: List[np.ndarray] = []

        sampled_frames = 0
        frame_iter_seconds = 0.0
        mouth_crop_seconds = 0.0
        landmark_seconds = 0.0
        loop_start = time.perf_counter()
        for gt_frame, inf_frame in self._iter_sampled_frame_pairs(gt_path, inf_path):
            sampled_frames += 1

            section_start = time.perf_counter()
            gt_crop = self._extract_mouth_crop(gt_frame)
            inf_crop = self._extract_mouth_crop(inf_frame)
            mouth_crop_seconds += time.perf_counter() - section_start
            if gt_crop is not None and inf_crop is not None:
                gt_lip_crops.append(gt_crop)
                inf_lip_crops.append(inf_crop)

            if self.landmarker is not None:
                section_start = time.perf_counter()
                gt_lm = self._extract_lip_landmarks(gt_frame)
                inf_lm = self._extract_lip_landmarks(inf_frame)
                landmark_seconds += time.perf_counter() - section_start
                if gt_lm is not None and inf_lm is not None:
                    gt_landmarks.append(gt_lm)
                    inf_landmarks.append(inf_lm)
        frame_iter_seconds = time.perf_counter() - loop_start - mouth_crop_seconds - landmark_seconds

        if sampled_frames == 0:
            LOGGER.info(
                "Timing | visual_run gt=%s inf=%s frames=0 total=%s",
                Path(gt_path).name,
                Path(inf_path).name,
                format_duration(time.perf_counter() - run_start),
            )
            return self._empty_result("empty_video")

        notes = []
        if not gt_lip_crops:
            LOGGER.info(
                "Timing | visual_run gt=%s inf=%s sampled_frames=%d mouth_crop=%s landmarks=%s frame_io_other=%s total=%s note=no_face_or_mouth_crops",
                Path(gt_path).name,
                Path(inf_path).name,
                sampled_frames,
                format_duration(mouth_crop_seconds),
                format_duration(landmark_seconds),
                format_duration(max(0.0, frame_iter_seconds)),
                format_duration(time.perf_counter() - run_start),
            )
            return self._empty_result("no_face_or_mouth_crops")

        section_start = time.perf_counter()
        gt_arr = np.stack(gt_lip_crops).astype(np.float32)
        inf_arr = np.stack(inf_lip_crops).astype(np.float32)
        diff = gt_arr - inf_arr
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        psnr = self._psnr(mse)
        array_metric_seconds = time.perf_counter() - section_start

        section_start = time.perf_counter()
        ssim = float(np.mean([self._ssim(a, b) for a, b in zip(gt_arr, inf_arr)]))
        jerk = self._lip_crop_jerk(inf_arr)
        temporal_metric_seconds = time.perf_counter() - section_start

        lde = math.nan
        landmark_metric_seconds = 0.0
        if gt_landmarks and inf_landmarks:
            section_start = time.perf_counter()
            gt_lm_arr = np.stack(gt_landmarks)
            inf_lm_arr = np.stack(inf_landmarks)
            lde = float(np.mean(np.linalg.norm(gt_lm_arr - inf_lm_arr, axis=2)))
            landmark_metric_seconds = time.perf_counter() - section_start
        else:
            notes.append("lde_unavailable_without_mediapipe_landmarks")

        notes.append("fid_requires_pretrained_inception_or_fvd_features")
        LOGGER.info(
            "Timing | visual_run gt=%s inf=%s sampled_frames=%d visual_frames=%d landmark_frames=%d frame_io_other=%s mouth_crop=%s landmark_detect=%s array_metrics=%s ssim_jerk=%s landmark_metric=%s total=%s",
            Path(gt_path).name,
            Path(inf_path).name,
            sampled_frames,
            len(gt_lip_crops),
            len(gt_landmarks),
            format_duration(max(0.0, frame_iter_seconds)),
            format_duration(mouth_crop_seconds),
            format_duration(landmark_seconds),
            format_duration(array_metric_seconds),
            format_duration(temporal_metric_seconds),
            format_duration(landmark_metric_seconds),
            format_duration(time.perf_counter() - run_start),
        )

        return {
            "lde": lde,
            "fid": math.nan,
            "jerk": jerk,
            "mouth_mae": mae,
            "mouth_mse": mse,
            "mouth_psnr": psnr,
            "mouth_ssim": ssim,
            "visual_frames": len(gt_lip_crops),
            "landmark_frames": len(gt_landmarks),
            "visual_note": ";".join(notes),
        }

    @staticmethod
    def _empty_result(note: str) -> dict:
        return {
            "lde": math.nan,
            "fid": math.nan,
            "jerk": math.nan,
            "mouth_mae": math.nan,
            "mouth_mse": math.nan,
            "mouth_psnr": math.nan,
            "mouth_ssim": math.nan,
            "visual_frames": 0,
            "landmark_frames": 0,
            "visual_note": note,
        }

    def _iter_sampled_frame_pairs(self, gt_path: str, inf_path: str):
        gt_cap = cv2.VideoCapture(gt_path)
        inf_cap = cv2.VideoCapture(inf_path)
        idx = 0
        try:
            while gt_cap.isOpened() and inf_cap.isOpened():
                if idx % self.frame_stride == 0:
                    gt_ok, gt_frame = gt_cap.read()
                    inf_ok, inf_frame = inf_cap.read()
                    if not gt_ok or not inf_ok:
                        break
                    yield gt_frame, inf_frame
                else:
                    gt_ok = gt_cap.grab()
                    inf_ok = inf_cap.grab()
                    if not gt_ok or not inf_ok:
                        break
                idx += 1
        finally:
            gt_cap.release()
            inf_cap.release()

    @staticmethod
    def _make_opencv_face_detector():
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade.exists():
            return None
        detector = cv2.CascadeClassifier(str(cascade))
        return detector if not detector.empty() else None

    def _extract_mouth_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        face = self._detect_largest_face(frame)
        if face is None:
            x, y, fw, fh = w // 4, h // 4, w // 2, h // 2
        else:
            x, y, fw, fh = face

        mx0 = max(0, x + int(0.18 * fw))
        mx1 = min(w, x + int(0.82 * fw))
        my0 = max(0, y + int(0.58 * fh))
        my1 = min(h, y + int(0.95 * fh))
        if mx1 <= mx0 or my1 <= my0:
            return None

        crop = frame[my0:my1, mx0:mx1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _detect_largest_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.face_detector is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda box: box[2] * box[3]))

    @staticmethod
    def _make_mediapipe_landmarker():
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception:
            return None

        model_path = Path(__file__).resolve().parents[1] / "src" / "utils" / "mp_models" / "face_landmarker_v2_with_blendshapes.task"
        if not model_path.exists():
            return None

        try:
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            return vision.FaceLandmarker.create_from_options(options)
        except Exception:
            return None

    def _extract_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.landmarker is None:
            return None
        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(image)
            if not result.face_landmarks:
                return None
            points = result.face_landmarks[0]
            lip = np.array([[points[i].x, points[i].y] for i in LIP_LANDMARKS], dtype=np.float32)
            center = lip.mean(axis=0, keepdims=True)
            scale = np.linalg.norm(lip.max(axis=0) - lip.min(axis=0))
            return (lip - center) / max(scale, 1e-6)
        except Exception:
            return None

    @staticmethod
    def _psnr(mse: float) -> float:
        if mse <= 1e-12:
            return float("inf")
        return float(20.0 * math.log10(1.0 / math.sqrt(mse)))

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        var_a = float(a.var())
        var_b = float(b.var())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _lip_crop_jerk(crops: np.ndarray) -> float:
        if len(crops) < 4:
            return math.nan
        motion = np.mean(np.abs(np.diff(crops, axis=0)), axis=(1, 2))
        jerk = np.diff(motion, n=3)
        return float(np.mean(np.abs(jerk))) if jerk.size else math.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic visual quality evaluation")
    parser.add_argument("--gt", required=True, help="Path to Ground Truth video")
    parser.add_argument("--pred", required=True, help="Path to Predicted video")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    evaluator = VisualEvaluator(frame_stride=args.frame_stride)
    res = evaluator.run(args.gt, args.pred)
    print(f"Visual Quality Result: {res}")
