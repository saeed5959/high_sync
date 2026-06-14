from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EvalResult:
    lse_d: Optional[float] = None
    lse_c: Optional[float] = None
    offset: Optional[int] = None
    lde: Optional[float] = None
    fid: Optional[float] = None
    jerk: Optional[float] = None
    mouth_mae: Optional[float] = None
    mouth_psnr: Optional[float] = None
    mouth_ssim: Optional[float] = None
    wer: Optional[float] = None

@dataclass
class InferenceItem:
    id: str
    video_path: str
    method_name: str     # e.g. "Ours", "Wav2Lip"
    audio_path: Optional[str] = None
    audio_source: Optional[str] = None
    result: EvalResult = field(default_factory=EvalResult)

@dataclass
class GroundTruthItem:
    id: str
    video_path: str
    transcript: str = ""
    inferences: List[InferenceItem] = field(default_factory=list)
