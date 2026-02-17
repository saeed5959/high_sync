from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EvalResult:
    lse_d: Optional[float] = None
    lse_c: Optional[float] = None
    offset: Optional[int] = None
    lde: Optional[float] = None
    fid: Optional[float] = None
    wer: Optional[float] = None

@dataclass
class InferenceItem:
    id: str
    video_path: str
    audio_path: str      # Path to extracted temporary wav
    method_name: str     # e.g. "Ours", "Wav2Lip"
    result: EvalResult = field(default_factory=EvalResult)

@dataclass
class GroundTruthItem:
    id: str
    video_path: str
    transcript: str = ""
    inferences: List[InferenceItem] = field(default_factory=list)