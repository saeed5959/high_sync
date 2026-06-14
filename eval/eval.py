import os
import subprocess
import shutil
import argparse
import logging
import json
import platform
import sys
import time
import hashlib
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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


LOGGER = logging.getLogger("highsync.eval")
EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DEFAULT_FID_INCEPTION_PATH = EVAL_ROOT / "pretrained" / "fid" / "inception_v3_google-0cc3c7bd.pth"
NUMERIC_METRIC_COLUMNS = [
    "LSE-D",
    "LSE-C",
    "Offset",
    "LDE",
    "FID",
    "Jerk",
    "Mouth-MAE",
    "Mouth-MSE",
    "Mouth-PSNR",
    "Mouth-SSIM",
    "VisualFrames",
    "LandmarkFrames",
    "WER",
]
EXPECTED_PRETRAINED = {
    "syncnet": {
        "path": EVAL_ROOT / "pretrained" / "syncnet" / "syncnet_v2.model",
        "required_for": "official SyncNet LSE-D/LSE-C/Offset",
        "min_bytes": 10_000_000,
        "expected_sha256": None,
    },
    "inception_v3": {
        "path": EVAL_ROOT / "pretrained" / "fid" / "inception_v3_google-0cc3c7bd.pth",
        "required_for": "FID",
        "min_bytes": 100_000_000,
        "expected_sha256": "0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef",
    },
    "i3d_kinetics": {
        "path": EVAL_ROOT / "pretrained" / "fvd" / "I3D_8x8_R50.pyth",
        "required_for": "FVD",
        "min_bytes": 100_000_000,
        "expected_sha256": None,
    },
    "mediapipe_face_landmarker": {
        "path": PROJECT_ROOT / "src" / "utils" / "mp_models" / "face_landmarker_v2_with_blendshapes.task",
        "required_for": "LDE",
        "min_bytes": 1_000_000,
        "expected_sha256": None,
    },
}


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class SimpleProgress:
    def __init__(self, total: int, desc: str = "Progress", unit: str = "item", disable: bool = False):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.current = 0
        self.postfix = ""
        if not self.disable:
            print(f"{self.desc}: 0/{self.total} {self.unit}")

    def set_postfix_str(self, text: str):
        self.postfix = text
        if not self.disable:
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} | {self.postfix}")

    def update(self, n: int = 1):
        self.current += n
        if not self.disable:
            print(f"{self.desc}: {self.current}/{self.total} {self.unit} | {self.postfix}")

    def close(self):
        if not self.disable:
            print(f"{self.desc}: done {self.current}/{self.total} {self.unit}")


def make_progress(total: int, desc: str, unit: str, disable: bool):
    if tqdm is None:
        return SimpleProgress(total=total, desc=desc, unit=unit, disable=disable)
    return tqdm(total=total, desc=desc, unit=unit, disable=disable)


def setup_logging(log_file: Optional[str], log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = TqdmLoggingHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def quiet_third_party_logs() -> None:
    # MediaPipe/Google C++ dependencies can emit Clearcut telemetry messages
    # directly to stderr. Keep benchmark logs focused unless low-level debugging
    # is explicitly requested.
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    logging.getLogger("absl").setLevel(logging.CRITICAL)


def select_device(requested_device: str = "auto") -> str:
    requested_device = requested_device.lower()
    if requested_device == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        if requested_device == "cuda":
            LOGGER.warning("CUDA was requested, but PyTorch is not installed. Falling back to CPU.")
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available:
        LOGGER.warning("CUDA was requested, but torch.cuda.is_available() is False. Falling back to CPU.")
        return "cpu"

    if cuda_available and requested_device in {"auto", "cuda"}:
        device = "cuda:0"
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        return device

    return "cpu"


def log_device(device: str) -> None:
    if device.startswith("cuda"):
        try:
            import torch

            index = torch.device(device).index or 0
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            LOGGER.info(
                "Using device: %s (%s, capability=%s, torch_cuda=%s)",
                device,
                name,
                capability,
                torch.version.cuda,
            )
            return
        except Exception:
            LOGGER.info("Using device: %s", device)
            return

    LOGGER.info("Using device: CPU")


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


def load_dataset(extractor=None, methods=None):
    root = Path(__file__).parent / "test_data"
    gt_dir = root / "ground_truth"
    inf_root = root / "inference"
    dataset = []
    gt_files = sorted(list(gt_dir.glob("*.mp4")))
    if not gt_files:
        LOGGER.warning("No ground-truth MP4 files found in %s", gt_dir)
        return []
    LOGGER.info("Found %d ground-truth videos in %s", len(gt_files), gt_dir)
    for gt_path in gt_files:
        gt_id = gt_path.stem
        transcript = gt_path.with_suffix(".txt").read_text().strip() if gt_path.with_suffix(".txt").exists() else ""
        gt_item = GroundTruthItem(id=gt_id, video_path=str(gt_path), transcript=transcript)
        if inf_root.exists():
            for method_dir in inf_root.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    if methods and method_name not in methods:
                        continue
                    for inf_file in method_dir.glob("*.mp4"):
                        inf_name = inf_file.stem
                        if (inf_name == gt_id) or (inf_name.startswith(f"{gt_id}_")):
                            gt_item.inferences.append(
                                InferenceItem(
                                    id=inf_name,
                                    video_path=str(inf_file),
                                    method_name=method_name,
                                )
                            )
        if extractor is not None and gt_item.inferences:
            audio_source = sorted(
                gt_item.inferences,
                key=lambda item: (item.id, item.method_name, item.video_path),
            )[0]
            shared_audio_path = extractor.extract(audio_source.video_path)
            if shared_audio_path:
                for inf in gt_item.inferences:
                    inf.audio_path = shared_audio_path
                    inf.audio_source = f"{audio_source.method_name}/{audio_source.id}"
                LOGGER.info(
                    "Using shared sync audio for %s from %s",
                    gt_id,
                    audio_source.video_path,
                )
            else:
                LOGGER.warning(
                    "Could not extract shared sync audio for %s from %s; sync will fall back to each output video audio",
                    gt_id,
                    audio_source.video_path,
                )
        dataset.append(gt_item)
    return dataset


def count_inferences(dataset, limit=None) -> int:
    total = sum(len(gt.inferences) for gt in dataset)
    return min(total, limit) if limit is not None else total


def make_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in NUMERIC_METRIC_COLUMNS if col in df.columns]
    grouped = df.groupby("Method")[metric_cols]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, "NumVideos", df.groupby("Method").size().reindex(summary["Method"]).to_numpy())
    return summary


def make_metric_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in NUMERIC_METRIC_COLUMNS:
        if metric not in df.columns:
            continue
        available = int(df[metric].notna().sum())
        total = int(len(df))
        rows.append(
            {
                "Metric": metric,
                "Available": available,
                "Missing": total - available,
                "Availability": available / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def natural_gt_key(gt_id: str):
    prefix = "".join(ch for ch in gt_id if not ch.isdigit())
    digits = "".join(ch for ch in gt_id if ch.isdigit())
    return prefix, int(digits or 0)


def compute_fid_report(args, results_df: pd.DataFrame, device: str) -> pd.DataFrame:
    try:
        from fid import (
            InceptionFeatureExtractor,
            common_ids,
            compute_method_fid,
            filter_ids,
            list_source_ids,
            make_face_detector,
            method_video_map,
            select_device as select_fid_device,
        )
    except Exception as exc:
        raise RuntimeError("Could not import FID evaluator. Check fid.py and dependencies.") from exc

    source_dir = EVAL_ROOT / "test_data" / "ground_truth"
    inference_root = EVAL_ROOT / "test_data" / "inference"
    methods = sorted(results_df["Method"].dropna().astype(str).unique())
    source_ids = list_source_ids(source_dir)
    method_maps = {method: method_video_map(inference_root, method) for method in methods}
    evaluated_ids_by_method = {
        method: sorted(
            set(results_df.loc[results_df["Method"] == method, "GT_ID"].astype(str)),
            key=natural_gt_key,
        )
        for method in methods
    }

    requested_ids = args.fid_gt_ids
    if args.fid_method_specific_ids:
        ids_by_method = {
            method: filter_ids(
                [gt_id for gt_id in evaluated_ids_by_method[method] if gt_id in set(source_ids) and gt_id in method_maps[method]],
                requested_ids,
                args.fid_limit_videos,
            )
            for method in methods
        }
    else:
        eligible_maps = {
            method: {
                gt_id: path
                for gt_id, path in method_maps[method].items()
                if gt_id in set(evaluated_ids_by_method[method])
            }
            for method in methods
        }
        ids = filter_ids(common_ids(source_ids, eligible_maps), requested_ids, args.fid_limit_videos)
        ids_by_method = {method: ids for method in methods}

    fid_device = select_fid_device(args.fid_device if args.fid_device != "eval" else ("cuda" if device.startswith("cuda") else "cpu"))
    detector = make_face_detector() if args.fid_crop_mode == "face" else None
    extractor = InceptionFeatureExtractor(Path(args.fid_inception_path), fid_device)
    real_stats_cache = {}

    fid_args = SimpleNamespace(
        frame_stride=args.fid_frame_stride,
        max_frames_per_video=args.fid_max_frames_per_video,
        crop_mode=args.fid_crop_mode,
        batch_size=args.fid_batch_size,
    )

    rows = []
    for method in methods:
        ids = ids_by_method[method]
        if not ids:
            rows.append(
                {
                    "Method": method,
                    "FID": float("nan"),
                    "SourceFrames": 0,
                    "GeneratedFrames": 0,
                    "NumVideos": 0,
                    "FrameStride": args.fid_frame_stride,
                    "CropMode": args.fid_crop_mode,
                    "IDs": "",
                }
            )
            continue
        LOGGER.info("Computing FID for %s on %d video(s)", method, len(ids))
        rows.append(
            compute_method_fid(
                method=method,
                ids=ids,
                source_dir=source_dir,
                method_videos=method_maps[method],
                extractor=extractor,
                detector=detector,
                args=fid_args,
                real_stats_cache=real_stats_cache,
            )
        )

    fid_df = pd.DataFrame(rows)
    fid_df.to_csv(args.fid_output, index=False)
    return fid_df


def module_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_benchmark_setup() -> int:
    rows = []
    failures = 0
    warnings = 0

    for name, spec in EXPECTED_PRETRAINED.items():
        path = spec["path"]
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status = "OK"
        note = ""
        digest = ""

        if not exists:
            status = "MISSING"
            note = f"needed for {spec['required_for']}"
            failures += 1
        elif size < spec["min_bytes"]:
            status = "WARNING"
            note = f"file is smaller than expected for {spec['required_for']}"
            warnings += 1
        elif spec["expected_sha256"]:
            digest = sha256_file(path)
            if digest.lower() != spec["expected_sha256"]:
                status = "WARNING"
                note = "checksum differs from expected official file"
                warnings += 1
        rows.append(
            {
                "Component": name,
                "Status": status,
                "Path": str(path),
                "SizeBytes": size,
                "SHA256": digest,
                "Note": note,
            }
        )

    meccano_files = list((EVAL_ROOT / "pretrained" / "fvd").glob("*MECCANO*.pyth"))
    for file_path in meccano_files:
        rows.append(
            {
                "Component": "i3d_meccano_suspicious",
                "Status": "WARNING",
                "Path": str(file_path),
                "SizeBytes": file_path.stat().st_size,
                "SHA256": "",
                "Note": "MECCANO checkpoint is not the standard Kinetics-400 I3D file for FVD",
            }
        )
        warnings += 1

    packages = {
        "torch": "FID/FVD/SyncNet model loading",
        "torchvision": "FID InceptionV3",
        "mediapipe": "LDE landmarks",
        "tqdm": "progress bars",
        "whisper": "ASR WER if --whisper-model is used",
    }
    for package, purpose in packages.items():
        version = module_version(package)
        if version is None and package in {"torch", "torchvision"}:
            status = "MISSING"
            failures += 1
        elif version is None:
            status = "WARNING"
            warnings += 1
        else:
            status = "OK"
        rows.append(
            {
                "Component": f"python_package:{package}",
                "Status": status,
                "Path": "",
                "SizeBytes": "",
                "SHA256": "",
                "Note": f"{version or 'not installed'}; used for {purpose}",
            }
        )

    df = pd.DataFrame(rows)
    print("\n=== Benchmark Setup Check ===")
    print(df.to_string(index=False))
    df.to_csv("benchmark_setup_check.csv", index=False)
    print("\nSaved setup report to benchmark_setup_check.csv")
    print(f"Summary: {failures} missing required items, {warnings} warnings")
    return 1 if failures else 0


def write_metadata(
    args,
    dataset,
    results_count: int,
    elapsed_seconds: float,
    summary_output: str,
    availability_output: str,
    metadata_output: str,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_type": "automatic_objective_metrics",
        "human_ui_included": False,
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "outputs": {
            "detailed_results": args.output,
            "method_summary": summary_output,
            "metric_availability": availability_output,
            "fid_summary": args.fid_output if getattr(args, "compute_fid", False) else None,
            "metadata": metadata_output,
            "log_file": args.log_file or None,
        },
        "configuration": {
            "methods": args.methods,
            "frame_stride": args.frame_stride,
            "limit": args.limit,
            "whisper_model": args.whisper_model,
            "device": args.device,
            "compute_fid": getattr(args, "compute_fid", False),
            "fid_frame_stride": getattr(args, "fid_frame_stride", None),
            "fid_crop_mode": getattr(args, "fid_crop_mode", None),
            "fid_common_ids_only": not getattr(args, "fid_method_specific_ids", False),
        },
        "dataset": {
            "ground_truth_videos": len(dataset),
            "matched_inference_videos": count_inferences(dataset, args.limit),
            "evaluated_videos": results_count,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "package_versions": {
            "pandas": module_version("pandas"),
            "numpy": module_version("numpy"),
            "opencv": module_version("cv2"),
            "soundfile": module_version("soundfile"),
            "scipy": module_version("scipy"),
            "tqdm": module_version("tqdm"),
            "mediapipe": module_version("mediapipe"),
            "torch": module_version("torch"),
        },
        "metric_notes": {
            "LSE-D/LSE-C/Offset": "Deterministic audio-video sync proxy from mouth-motion/audio-energy cross-correlation. When available, all methods for the same GT_ID are evaluated against one shared audio track extracted from an available inference output. This is not official SyncNet unless replaced with a pretrained SyncNet implementation.",
            "Mouth-MAE/MSE/PSNR/SSIM": "Deterministic lower-mouth crop reconstruction metrics against the matching source/ground-truth video. These are not lip-correctness metrics when source speech and driving audio differ.",
            "Jerk": "Temporal smoothness proxy from lower-mouth crop motion.",
            "LDE": "MediaPipe lip landmark distance when MediaPipe is installed and landmarks are detected; otherwise NaN.",
            "FID": "Method-level Frechet Inception Distance over sampled video frames. Lower is better. When --compute-fid is enabled, the method-level value is repeated in final_results.csv and written separately to fid_summary.csv.",
            "WER": "NaN unless reference and predicted transcripts or an explicit ASR model are available.",
        },
    }
    Path(metadata_output).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ==========================================
# Main Logic (Updated)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for talking-head videos")
    parser.add_argument(
        "--check-benchmark-setup",
        action="store_true",
        help="Validate pretrained files and Python packages, then exit.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method folder names under test_data/inference, e.g. ours wav2lip",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame for visual metrics. Use 1 for final paper numbers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of inference videos for smoke testing.",
    )
    parser.add_argument(
        "--output",
        default="final_results.csv",
        help="CSV file to write.",
    )
    parser.add_argument(
        "--summary-output",
        default="benchmark_summary.csv",
        help="Per-method aggregate benchmark CSV to write.",
    )
    parser.add_argument(
        "--availability-output",
        default="metric_availability.csv",
        help="Metric availability report CSV to write.",
    )
    parser.add_argument(
        "--metadata-output",
        default="benchmark_metadata.json",
        help="Reproducibility metadata JSON to write.",
    )
    parser.add_argument(
        "--compute-fid",
        action="store_true",
        help="Compute method-level FID and fill the FID column in the result CSV.",
    )
    parser.add_argument(
        "--fid-output",
        default="fid_summary.csv",
        help="Method-level FID report CSV to write when --compute-fid is enabled.",
    )
    parser.add_argument(
        "--fid-inception-path",
        default=str(DEFAULT_FID_INCEPTION_PATH),
        help="Path to the pretrained InceptionV3 checkpoint for FID.",
    )
    parser.add_argument(
        "--fid-frame-stride",
        type=int,
        default=10,
        help="Use every Nth frame for FID frame sampling.",
    )
    parser.add_argument(
        "--fid-max-frames-per-video",
        type=int,
        default=None,
        help="Optional max FID frames sampled from each video.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=32,
        help="Batch size for InceptionV3 FID feature extraction.",
    )
    parser.add_argument(
        "--fid-crop-mode",
        choices=["face", "full"],
        default="face",
        help="FID crop mode. Use face for talking-head visual quality.",
    )
    parser.add_argument(
        "--fid-method-specific-ids",
        action="store_true",
        help="Compute each method's FID on its available evaluated IDs. Default uses IDs common to all evaluated methods.",
    )
    parser.add_argument(
        "--fid-gt-ids",
        nargs="*",
        default=None,
        help="Optional GT_ID subset for FID, e.g. s1 s12.",
    )
    parser.add_argument(
        "--fid-limit-videos",
        type=int,
        default=None,
        help="Optional number of GT_IDs to use for FID after filtering.",
    )
    parser.add_argument(
        "--fid-device",
        choices=["eval", "auto", "cuda", "cpu"],
        default="eval",
        help="Device for FID. eval reuses the eval.py device choice.",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Optional Whisper model name/path for ASR-based WER, e.g. medium.en or large.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for model-based metrics. auto uses CUDA when PyTorch can see a GPU.",
    )
    parser.add_argument(
        "--log-file",
        default="evaluation.log",
        help="Write detailed logs to this file. Use an empty string to disable file logging.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    parser.add_argument(
        "--show-third-party-logs",
        action="store_true",
        help="Show verbose logs from MediaPipe/TensorFlow/Google dependencies.",
    )
    args = parser.parse_args()
    if not args.show_third_party_logs:
        quiet_third_party_logs()
    setup_logging(args.log_file or None, args.log_level)
    if args.check_benchmark_setup:
        raise SystemExit(check_benchmark_setup())
    device = select_device(args.device)
    log_device(device)
    perf_start = time.perf_counter()

    extractor = AudioExtractor()

    try:
        # 1. Load Data
        section_start = time.perf_counter()
        dataset = load_dataset(extractor, methods=set(args.methods) if args.methods else None)
        LOGGER.info("Timing | load_dataset=%s", format_duration(time.perf_counter() - section_start))
        if not dataset:
            LOGGER.error("Dataset is empty. Nothing to evaluate.")
            return

        # 2. Init Models
        LOGGER.info("Initializing evaluators")
        section_start = time.perf_counter()
        sync_eval = SyncEvaluator()
        LOGGER.info("Timing | init_sync=%s", format_duration(time.perf_counter() - section_start))
        section_start = time.perf_counter()
        vis_eval = VisualEvaluator(device=device, frame_stride=args.frame_stride)
        LOGGER.info("Timing | init_visual=%s", format_duration(time.perf_counter() - section_start))
        section_start = time.perf_counter()
        sem_eval = SemanticEvaluator(whisper_model_name=args.whisper_model, device=device)
        LOGGER.info("Timing | init_semantic=%s", format_duration(time.perf_counter() - section_start))

        results = []

        # 3. Process All Videos
        section_start = time.perf_counter()
        total_inferences = count_inferences(dataset, args.limit)
        LOGGER.info("Timing | count_inferences=%s", format_duration(time.perf_counter() - section_start))
        if total_inferences == 0:
            LOGGER.error("No matching inference videos found.")
            return

        LOGGER.info(
            "Starting evaluation: %d videos, frame_stride=%d, output=%s",
            total_inferences,
            args.frame_stride,
            args.output,
        )
        processed = 0
        stop = False
        progress = make_progress(
            total=total_inferences,
            desc="Evaluating videos",
            unit="video",
            disable=args.no_progress,
        )
        for gt in dataset:
            if args.limit is not None and processed >= args.limit:
                break
            if not gt.inferences:
                LOGGER.debug("Skipping %s because no inference videos matched", gt.id)
                continue

            LOGGER.info("Processing GT %s with %d matching inference videos", gt.id, len(gt.inferences))

            for inf in gt.inferences:
                if args.limit is not None and processed >= args.limit:
                    stop = True
                    break
                video_start = time.perf_counter()
                progress.set_postfix_str(f"{inf.method_name}/{inf.id}: sync")
                LOGGER.info("Evaluating %s/%s: sync", inf.method_name, inf.id)
                section_start = time.perf_counter()
                try:
                    s_res = sync_eval.run(inf.video_path, inf.audio_path)
                except Exception:
                    LOGGER.exception("Sync evaluation failed for %s", inf.video_path)
                    s_res = {"lse_d": 99.0, "lse_c": 0.0, "offset": 0}
                sync_seconds = time.perf_counter() - section_start
                LOGGER.info("Timing | %s/%s sync=%s", inf.method_name, inf.id, format_duration(sync_seconds))

                progress.set_postfix_str(f"{inf.method_name}/{inf.id}: visual")
                LOGGER.info("Evaluating %s/%s: visual", inf.method_name, inf.id)
                section_start = time.perf_counter()
                try:
                    v_res = vis_eval.run(gt.video_path, inf.video_path)
                except Exception:
                    LOGGER.exception("Visual evaluation failed for %s", inf.video_path)
                    v_res = VisualEvaluator._empty_result("visual_exception")
                visual_seconds = time.perf_counter() - section_start
                LOGGER.info("Timing | %s/%s visual=%s", inf.method_name, inf.id, format_duration(visual_seconds))

                progress.set_postfix_str(f"{inf.method_name}/{inf.id}: semantic")
                LOGGER.info("Evaluating %s/%s: semantic", inf.method_name, inf.id)
                section_start = time.perf_counter()
                try:
                    sem_res = sem_eval.run(gt.transcript, inf.video_path)
                except Exception:
                    LOGGER.exception("Semantic evaluation failed for %s", inf.video_path)
                    sem_res = {
                        "wer": float("nan"),
                        "pred_text": "",
                        "semantic_note": "semantic_exception",
                    }
                semantic_seconds = time.perf_counter() - section_start
                LOGGER.info("Timing | %s/%s semantic=%s", inf.method_name, inf.id, format_duration(semantic_seconds))

                # Append the results, including the new 'Jerk' metric
                results.append({
                    "Method": inf.method_name,
                    "GT_ID": gt.id,
                    "Inf_ID": inf.id,
                    "SyncAudioSource": inf.audio_source or "",
                    "LSE-D": s_res['lse_d'],
                    "LSE-C": s_res['lse_c'],
                    "Offset": s_res['offset'],
                    "LDE": v_res['lde'],
                    "FID": v_res['fid'],
                    "Jerk": v_res['jerk'],
                    "Mouth-MAE": v_res["mouth_mae"],
                    "Mouth-MSE": v_res["mouth_mse"],
                    "Mouth-PSNR": v_res["mouth_psnr"],
                    "Mouth-SSIM": v_res["mouth_ssim"],
                    "VisualFrames": v_res["visual_frames"],
                    "LandmarkFrames": v_res["landmark_frames"],
                    "WER": sem_res['wer'],
                    "VisualNote": v_res["visual_note"],
                    "SemanticNote": sem_res["semantic_note"],
                })
                processed += 1
                progress.update(1)
                progress.set_postfix_str(f"{inf.method_name}/{inf.id}: done")
                total_video_seconds = time.perf_counter() - video_start
                LOGGER.info(
                    "Timing | %s/%s total=%s breakdown(sync=%s visual=%s semantic=%s)",
                    inf.method_name,
                    inf.id,
                    format_duration(total_video_seconds),
                    format_duration(sync_seconds),
                    format_duration(visual_seconds),
                    format_duration(semantic_seconds),
                )
                LOGGER.info("Finished %s/%s (%d/%d)", inf.method_name, inf.id, processed, total_inferences)
            if stop:
                break
        progress.close()

        # 4. Save and Report
        if results:
            df = pd.DataFrame(results)
            if args.compute_fid:
                fid_start = time.perf_counter()
                LOGGER.info("Starting FID evaluation")
                try:
                    fid_df = compute_fid_report(args, df, device)
                    fid_by_method = fid_df.set_index("Method")["FID"].to_dict()
                    df["FID"] = df["Method"].map(fid_by_method)
                    if "VisualNote" in df.columns:
                        df["VisualNote"] = (
                            df["VisualNote"]
                            .fillna("")
                            .str.replace("fid_requires_pretrained_inception_or_fvd_features", "method_level_fid_computed", regex=False)
                            .str.strip(";")
                        )
                    LOGGER.info(
                        "Timing | fid=%s output=%s",
                        format_duration(time.perf_counter() - fid_start),
                        args.fid_output,
                    )
                    print("\n=== Method FID ===")
                    print(fid_df.drop(columns=["IDs"], errors="ignore").to_string(index=False))
                except Exception:
                    LOGGER.exception("FID evaluation failed; leaving FID as NaN")
            summary_df = make_method_summary(df)
            availability_df = make_metric_availability(df)
            LOGGER.info("Completed evaluation for %d videos", len(results))
            print("\n=== Method Averages ===")
            print(df.groupby("Method").mean(numeric_only=True))

            df.to_csv(args.output, index=False)
            summary_df.to_csv(args.summary_output, index=False)
            availability_df.to_csv(args.availability_output, index=False)
            write_metadata(
                args=args,
                dataset=dataset,
                results_count=len(results),
                elapsed_seconds=time.perf_counter() - perf_start,
                summary_output=args.summary_output,
                availability_output=args.availability_output,
                metadata_output=args.metadata_output,
            )
            LOGGER.info("Saved detailed results to %s", args.output)
            LOGGER.info("Saved method summary to %s", args.summary_output)
            LOGGER.info("Saved metric availability report to %s", args.availability_output)
            LOGGER.info("Saved benchmark metadata to %s", args.metadata_output)
        else:
            LOGGER.warning("No matching inferences found.")

    finally:
        # Clean up temp wav files
        cleanup_start = time.perf_counter()
        extractor.cleanup()
        LOGGER.info("Timing | cleanup=%s", format_duration(time.perf_counter() - cleanup_start))
        LOGGER.info("Timing | total_elapsed=%s", format_duration(time.perf_counter() - perf_start))
        LOGGER.info("Cleaned temporary audio cache")


if __name__ == "__main__":
    main()
