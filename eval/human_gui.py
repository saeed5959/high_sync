"""
PyQt5 Human Perception Evaluation Tool
Pleasant, resumable GUI for MOS and A/B lip-sync studies.
"""

import json
import random
import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class MOSResult:
    video: str
    method: str
    clip_id: str
    mos_sync: int
    mos_realism: int


@dataclass
class ABResult:
    video_a: str
    video_b: str
    method_a: str
    method_b: str
    clip_id: str
    preference: str  # 'A', 'B', or 'N'


class DataManager:
    def __init__(self, mos_path: Path, ab_path: Path):
        self.mos_path = mos_path
        self.ab_path = ab_path

    def load_mos(self) -> List[MOSResult]:
        if not self.mos_path.exists():
            return []
        return [MOSResult(**row) for row in json.loads(self.mos_path.read_text())]

    def save_mos(self, rows: List[MOSResult]) -> None:
        self.mos_path.write_text(json.dumps([asdict(r) for r in rows], indent=4))

    def load_ab(self) -> List[ABResult]:
        if not self.ab_path.exists():
            return []
        return [ABResult(**row) for row in json.loads(self.ab_path.read_text())]

    def save_ab(self, rows: List[ABResult]) -> None:
        self.ab_path.write_text(json.dumps([asdict(r) for r in rows], indent=4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect_inference_videos(root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    if not root.exists():
        return out
    for method_dir in root.iterdir():
        if method_dir.is_dir():
            videos = sorted(method_dir.glob("*.mp4"))
            if videos:
                out[method_dir.name] = videos
    return out


def make_ab_pairs(method_videos: Dict[str, List[Path]]) -> List[Dict[str, Path]]:
    pairs = []
    for method_a, method_b in combinations(method_videos.keys(), 2):
        files_a = {p.name: p for p in method_videos[method_a]}
        files_b = {p.name: p for p in method_videos[method_b]}
        overlap = files_a.keys() & files_b.keys()
        for name in overlap:
            a, b = files_a[name], files_b[name]
            pair = {"A": a, "B": b} if random.random() > 0.5 else {"A": b, "B": a}
            pairs.append(pair)
    random.shuffle(pairs)
    return pairs


def ordered_ab_key(method_a: str, method_b: str, clip_id: str) -> Tuple[str, str, str]:
    a, b = sorted([method_a, method_b])
    return (a, b, clip_id)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class HumanEvalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Perception Evaluation Tool")
        self.resize(1280, 820)

        self.root = Path(__file__).parent
        self.inf_root = self.root / "test_data" / "inference"
        self.dm = DataManager(
            self.root / "human_eval_mos.json",
            self.root / "human_eval_ab_test.json",
        )

        self.mos_results = self.dm.load_mos()
        self.ab_results = self.dm.load_ab()

        self.mos_queue: List[Path] = []
        self.mos_index: int = -1

        self.ab_pairs: List[Dict[str, Path]] = []
        self.ab_index: int = -1

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_mos_tab()
        self._init_ab_tab()
        self._init_results_tab()

    # ------------------------------------------------------------------
    # Player factory with diagnostics
    # ------------------------------------------------------------------
    def _create_player(self, label: QLabel) -> QMediaPlayer:
        player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        player.mediaStatusChanged.connect(
            lambda status, lbl=label, pl=player: self._handle_media_status(pl, status, lbl)
        )
        player.error.connect(
            lambda code, lbl=label, pl=player: self._handle_media_error(pl, code, lbl)
        )
        player.setNotifyInterval(100)
        player.setVolume(100)
        return player

    @staticmethod
    def _handle_media_status(player: QMediaPlayer, status: QMediaPlayer.MediaStatus, label: QLabel):
        if status == QMediaPlayer.LoadedMedia:
            player.play()
        elif status == QMediaPlayer.InvalidMedia:
            label.setText(f"[Error] Cannot load media: {player.media().canonicalUrl().toLocalFile()}")
        elif status == QMediaPlayer.BufferingMedia:
            label.setText("Buffering...")
        elif status == QMediaPlayer.EndOfMedia:
            label.setText("Playback finished.")
        elif status == QMediaPlayer.LoadingMedia:
            label.setText("Loading...")

    @staticmethod
    def _handle_media_error(player: QMediaPlayer, code: QMediaPlayer.Error, label: QLabel):
        if code == QMediaPlayer.NoError:
            return
        msg = (
            f"[Playback Error] {player.errorString()} "
            f"({player.media().canonicalUrl().toLocalFile()})"
        )
        label.setText(msg)
        QMessageBox.critical(label.window(), "Video Playback Error", msg)

    # ------------------------------------------------------------------
    # MOS TAB
    # ------------------------------------------------------------------
    def _init_mos_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.mos_video_widget = QVideoWidget()
        self.mos_status = QLabel("Click “Start MOS Study” to begin.")
        self.mos_status.setWordWrap(True)

        self.mos_player = self._create_player(self.mos_status)
        self.mos_player.setVideoOutput(self.mos_video_widget)

        layout.addWidget(self.mos_video_widget, 1)
        layout.addWidget(self.mos_status)

        controls = QHBoxLayout()
        self.mos_start_btn = QPushButton("Start MOS Study")
        self.mos_start_btn.clicked.connect(self.start_mos)
        controls.addWidget(self.mos_start_btn)

        self.mos_replay_btn = QPushButton("Replay")
        self.mos_replay_btn.clicked.connect(lambda: self.mos_player.play())
        controls.addWidget(self.mos_replay_btn)

        controls.addStretch()
        controls.addWidget(QLabel("Lip-Sync (1–5):"))
        self.mos_sync_spin = QSpinBox()
        self.mos_sync_spin.setRange(1, 5)
        self.mos_sync_spin.setValue(3)
        controls.addWidget(self.mos_sync_spin)

        controls.addWidget(QLabel("Realism (1–5):"))
        self.mos_realism_spin = QSpinBox()
        self.mos_realism_spin.setRange(1, 5)
        self.mos_realism_spin.setValue(3)
        controls.addWidget(self.mos_realism_spin)

        self.mos_submit_btn = QPushButton("Submit & Next")
        self.mos_submit_btn.clicked.connect(self.submit_mos)
        controls.addWidget(self.mos_submit_btn)

        layout.addLayout(controls)
        self.tabs.addTab(tab, "MOS Study")
        self._set_mos_enabled(False)

    def _set_mos_enabled(self, enabled: bool):
        self.mos_sync_spin.setEnabled(enabled)
        self.mos_realism_spin.setEnabled(enabled)
        self.mos_submit_btn.setEnabled(enabled)
        self.mos_replay_btn.setEnabled(enabled)
        self.mos_start_btn.setEnabled(not enabled)

    def start_mos(self):
        videos = collect_inference_videos(self.inf_root)
        flat = [v for vs in videos.values() for v in vs]
        if not flat:
            QMessageBox.warning(self, "No Videos", "No MP4s under inference/.")
            return

        completed = {item.video for item in self.mos_results}
        remaining = [v for v in flat if str(v) not in completed]
        if not remaining:
            QMessageBox.information(self, "Complete", "All clips already rated.")
            return

        random.shuffle(remaining)
        self.mos_queue = remaining
        self.mos_index = 0
        self._load_mos_clip()
        self._set_mos_enabled(True)

    def _load_mos_clip(self):
        clip = self.mos_queue[self.mos_index]
        self.mos_status.setText(f"Video {self.mos_index + 1}/{len(self.mos_queue)}\n{clip}")
        self.mos_player.stop()
        self.mos_player.setMedia(QMediaContent(QUrl.fromLocalFile(str(clip.resolve()))))

    def submit_mos(self):
        clip = self.mos_queue[self.mos_index]
        result = MOSResult(
            video=str(clip),
            method=clip.parent.name,
            clip_id=clip.stem,
            mos_sync=int(self.mos_sync_spin.value()),
            mos_realism=int(self.mos_realism_spin.value()),
        )
        self.mos_results.append(result)
        self.dm.save_mos(self.mos_results)

        self.mos_index += 1
        if self.mos_index < len(self.mos_queue):
            self.mos_sync_spin.setValue(3)
            self.mos_realism_spin.setValue(3)
            self._load_mos_clip()
        else:
            QMessageBox.information(self, "Finished", "MOS study complete.")
            self.mos_player.stop()
            self._set_mos_enabled(False)
            self.mos_status.setText("All MOS clips finished. Reload to continue later.")
            self.refresh_tables()

    # ------------------------------------------------------------------
    # AB TAB
    # ------------------------------------------------------------------
    def _init_ab_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Video players side by side
        players_layout = QHBoxLayout()
        self.ab_status = QLabel("Click “Start A/B Study”.")
        self.ab_status.setWordWrap(True)

        self.ab_label_a = QLabel("-")
        self.ab_label_a.setWordWrap(True)
        self.ab_label_b = QLabel("-")
        self.ab_label_b.setWordWrap(True)

        video_widget_a = QVideoWidget()
        video_widget_b = QVideoWidget()

        self.ab_player_a = self._create_player(self.ab_label_a)
        self.ab_player_b = self._create_player(self.ab_label_b)
        self.ab_player_a.setVideoOutput(video_widget_a)
        self.ab_player_b.setVideoOutput(video_widget_b)

        left = QVBoxLayout()
        left.addWidget(QLabel("Video A"))
        left.addWidget(video_widget_a)
        left.addWidget(self.ab_label_a)
        right = QVBoxLayout()
        right.addWidget(QLabel("Video B"))
        right.addWidget(video_widget_b)
        right.addWidget(self.ab_label_b)
        players_layout.addLayout(left)
        players_layout.addLayout(right)

        layout.addLayout(players_layout)
        layout.addWidget(self.ab_status)

        controls = QHBoxLayout()
        self.ab_start_btn = QPushButton("Start A/B Study")
        self.ab_start_btn.clicked.connect(self.start_ab)
        controls.addWidget(self.ab_start_btn)

        self.ab_replay_a = QPushButton("Replay A")
        self.ab_replay_a.clicked.connect(lambda: self.ab_player_a.play())
        controls.addWidget(self.ab_replay_a)

        self.ab_replay_b = QPushButton("Replay B")
        self.ab_replay_b.clicked.connect(lambda: self.ab_player_b.play())
        controls.addWidget(self.ab_replay_b)

        controls.addStretch()
        self.ab_btn_a = QPushButton("Prefer A")
        self.ab_btn_b = QPushButton("Prefer B")
        self.ab_btn_n = QPushButton("Neither / Equal")
        self.ab_btn_a.clicked.connect(lambda: self.submit_ab("A"))
        self.ab_btn_b.clicked.connect(lambda: self.submit_ab("B"))
        self.ab_btn_n.clicked.connect(lambda: self.submit_ab("N"))
        controls.addWidget(self.ab_btn_a)
        controls.addWidget(self.ab_btn_b)
        controls.addWidget(self.ab_btn_n)

        layout.addLayout(controls)
        self.tabs.addTab(tab, "A/B Preference Test")
        self._set_ab_enabled(False)

    def _set_ab_enabled(self, enabled: bool):
        for btn in (self.ab_btn_a, self.ab_btn_b, self.ab_btn_n, self.ab_replay_a, self.ab_replay_b):
            btn.setEnabled(enabled)
        self.ab_start_btn.setEnabled(not enabled)

    def start_ab(self):
        methods = collect_inference_videos(self.inf_root)
        if len(methods) < 2:
            QMessageBox.warning(self, "Need ≥2 methods", "Create at least two folders in inference/.")
            return

        all_pairs = make_ab_pairs(methods)
        if not all_pairs:
            QMessageBox.warning(self, "No overlap", "No matching filenames across methods.")
            return

        completed = {
            ordered_ab_key(row.method_a, row.method_b, row.clip_id)
            for row in self.ab_results
        }
        remaining = [
            pair for pair in all_pairs
            if ordered_ab_key(pair["A"].parent.name, pair["B"].parent.name, pair["A"].stem) not in completed
        ]
        if not remaining:
            QMessageBox.information(self, "Finished", "All A/B pairs already rated.")
            return

        self.ab_pairs = remaining
        self.ab_index = 0
        self._load_ab_pair()
        self._set_ab_enabled(True)

    def _load_ab_pair(self):
        pair = self.ab_pairs[self.ab_index]
        a, b = pair["A"], pair["B"]
        self.ab_status.setText(f"Pair {self.ab_index + 1}/{len(self.ab_pairs)}")
        self.ab_label_a.setText(str(a))
        self.ab_label_b.setText(str(b))

        self.ab_player_a.stop()
        self.ab_player_b.stop()
        self.ab_player_a.setMedia(QMediaContent(QUrl.fromLocalFile(str(a.resolve()))))
        self.ab_player_b.setMedia(QMediaContent(QUrl.fromLocalFile(str(b.resolve()))))

    def submit_ab(self, pref: str):
        pair = self.ab_pairs[self.ab_index]
        a, b = pair["A"], pair["B"]
        result = ABResult(
            video_a=str(a),
            video_b=str(b),
            method_a=a.parent.name,
            method_b=b.parent.name,
            clip_id=a.stem,
            preference=pref,
        )
        self.ab_results.append(result)
        self.dm.save_ab(self.ab_results)

        self.ab_index += 1
        if self.ab_index < len(self.ab_pairs):
            self._load_ab_pair()
        else:
            QMessageBox.information(self, "Finished", "A/B study complete.")
            self.ab_player_a.stop()
            self.ab_player_b.stop()
            self._set_ab_enabled(False)
            self.ab_status.setText("A/B study done. Reload to continue later.")
            self.refresh_tables()

    # ------------------------------------------------------------------
    # RESULTS TAB
    # ------------------------------------------------------------------
    def _init_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        refresh_btn = QPushButton("Load / Refresh Results")
        refresh_btn.clicked.connect(self.refresh_tables)
        layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("MOS Results"))
        self.mos_table = QTableWidget()
        layout.addWidget(self.mos_table)

        layout.addWidget(QLabel("A/B Results"))
        self.ab_table = QTableWidget()
        layout.addWidget(self.ab_table)

        self.tabs.addTab(tab, "View Results")
        self.refresh_tables()

    def refresh_tables(self):
        self.mos_results = self.dm.load_mos()
        self.ab_results = self.dm.load_ab()

        if self.mos_results:
            mos_df = pd.DataFrame([asdict(r) for r in self.mos_results])
            self._populate_table(self.mos_table, mos_df)
        else:
            self.mos_table.clear()
            self.mos_table.setRowCount(0)
            self.mos_table.setColumnCount(0)

        if self.ab_results:
            ab_df = pd.DataFrame([asdict(r) for r in self.ab_results])
            self._populate_table(self.ab_table, ab_df)
        else:
            self.ab_table.clear()
            self.ab_table.setRowCount(0)
            self.ab_table.setColumnCount(0)

    @staticmethod
    def _populate_table(table: QTableWidget, df: pd.DataFrame):
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns)
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.iloc[r, c]))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    window = HumanEvalGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()