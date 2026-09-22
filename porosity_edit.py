"""
Manual pore-label editor for lunar-soil CT porosity sessions.

Typical workflow (GUI):
  python porosity_edit.py
    → pick dataset, tune threshold mix with live preview
    → Initialize edit session
    → paint / save / confirm

Resume an existing session:
  python porosity_edit.py --session "_edit_sessions/6-11"

Display: blue / yellow translucent pore shadows (no pink).
Labels you can paint:
  0 background | 1 solid | 2 closed pore | 3 open pore | 4 wrap/ignore
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

import porosity_ct as pc

# Pore shadows only: blue closed, yellow open (wrap not drawn)
LABEL_COLORS = {
    pc.L_BG: (0.0, 0.0, 0.0, 0.0),
    pc.L_SOLID: (0.0, 0.0, 0.0, 0.0),  # keep CT readable
    pc.L_CLOSED: (0.00, 0.45, 0.70, 0.55),
    pc.L_OPEN: (0.95, 0.78, 0.10, 0.55),
    pc.L_WRAP: (0.0, 0.0, 0.0, 0.0),  # hidden (no pink)
}

LABEL_BUTTONS = [
    (pc.L_SOLID, "1 Solid", "gray"),
    (pc.L_CLOSED, "2 Closed pore (blue)", "#0072B2"),
    (pc.L_OPEN, "3 Open pore (yellow)", "#F0C400"),
    (pc.L_WRAP, "4 Wrap/ignore (hidden)", "#666666"),
    (pc.L_BG, "0 Background", "#333333"),
]


def _overlay_rgba(labels_z: np.ndarray, alpha_scale: float = 1.0) -> np.ndarray:
    h, w = labels_z.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for code, rgba_c in LABEL_COLORS.items():
        if rgba_c[3] <= 0:
            continue
        m = labels_z == code
        if not m.any():
            continue
        rgba[m] = rgba_c
        rgba[m, 3] *= alpha_scale
    return rgba


def _disk_mask(h: int, w: int, y: float, x: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r * r


def _polygon_mask(h: int, w: int, verts_yx: list[tuple[float, float]]) -> np.ndarray:
    if len(verts_yx) < 3:
        return np.zeros((h, w), dtype=bool)
    from skimage.draw import polygon as sk_polygon

    ys = np.array([v[0] for v in verts_yx], dtype=np.float64)
    xs = np.array([v[1] for v in verts_yx], dtype=np.float64)
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def _rect_mask(h: int, w: int, y0: float, x0: float, y1: float, x1: float) -> np.ndarray:
    ya, yb = sorted((int(round(y0)), int(round(y1))))
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya = int(np.clip(ya, 0, h - 1))
    yb = int(np.clip(yb, 0, h - 1))
    xa = int(np.clip(xa, 0, w - 1))
    xb = int(np.clip(xb, 0, w - 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[ya : yb + 1, xa : xb + 1] = True
    return mask


class SliceCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7.5, 7.5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._ct = None
        self._overlay = None
        self._brush_artist = None
        self._lasso_line = None
        self._rect_artist = None
        self.tool = "brush"  # brush | lasso | rect
        self.paint_callback = None  # brush: (y, x, first, erase)
        self.region_callback = None  # lasso/rect: (mask, erase)
        self.hover_callback = None
        self.wheel_callback = None
        self._painting = False
        self._erase_drag = False
        self._lasso_pts: list[tuple[float, float]] = []
        self._rect_start: tuple[float, float] | None = None
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("scroll_event", self._on_scroll)

    def show_slice(self, ct: np.ndarray, labels_z: np.ndarray, title: str, opacity: float):
        lo, hi = np.percentile(ct[ct > 0], [1, 99]) if np.any(ct > 0) else (0, 1)
        rgba = _overlay_rgba(labels_z, opacity)
        if self._ct is None:
            self._ct = self.ax.imshow(ct, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            self._overlay = self.ax.imshow(rgba, interpolation="nearest")
        else:
            self._ct.set_data(ct)
            self._ct.set_clim(lo, hi)
            self._overlay.set_data(rgba)
        self.ax.set_title(title, fontsize=10)
        self.draw_idle()

    def clear_tool_artists(self):
        if self._brush_artist is not None:
            self._brush_artist.remove()
            self._brush_artist = None
        if self._lasso_line is not None:
            self._lasso_line.remove()
            self._lasso_line = None
        if self._rect_artist is not None:
            self._rect_artist.remove()
            self._rect_artist = None

    def set_brush_cursor(self, y: float | None, x: float | None, radius: float):
        if self.tool != "brush":
            if self._brush_artist is not None:
                self._brush_artist.remove()
                self._brush_artist = None
            return
        if self._brush_artist is not None:
            self._brush_artist.remove()
            self._brush_artist = None
        if y is None or x is None:
            self.draw_idle()
            return
        from matplotlib.patches import Circle

        self._brush_artist = Circle(
            (x, y), radius, fill=False, ec="lime", lw=1.2, alpha=0.9
        )
        self.ax.add_patch(self._brush_artist)
        self.draw_idle()

    def _event_yx(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        return float(event.ydata), float(event.xdata)

    def _update_lasso_line(self):
        if self._lasso_line is not None:
            self._lasso_line.remove()
            self._lasso_line = None
        if len(self._lasso_pts) < 2:
            self.draw_idle()
            return
        xs = [p[1] for p in self._lasso_pts] + [self._lasso_pts[0][1]]
        ys = [p[0] for p in self._lasso_pts] + [self._lasso_pts[0][0]]
        (self._lasso_line,) = self.ax.plot(xs, ys, color="lime", lw=1.4, alpha=0.95)
        self.draw_idle()

    def _update_rect(self, y: float, x: float):
        if self._rect_artist is not None:
            self._rect_artist.remove()
            self._rect_artist = None
        if self._rect_start is None:
            return
        from matplotlib.patches import Rectangle

        y0, x0 = self._rect_start
        self._rect_artist = Rectangle(
            (min(x0, x), min(y0, y)),
            abs(x - x0),
            abs(y - y0),
            fill=False,
            ec="lime",
            lw=1.4,
            ls="--",
        )
        self.ax.add_patch(self._rect_artist)
        self.draw_idle()

    def _on_press(self, event):
        if event.button not in (1, 3):
            return
        yx = self._event_yx(event)
        if yx is None:
            return
        erase = event.button == 3 or self._erase_drag
        if self.tool == "brush":
            self._painting = True
            self._erase_drag = erase
            if self.paint_callback:
                self.paint_callback(yx[0], yx[1], first=True, erase=erase)
            return
        if self.tool == "lasso":
            self._painting = True
            self._erase_drag = erase
            self._lasso_pts = [yx]
            self._update_lasso_line()
            return
        if self.tool == "rect":
            self._painting = True
            self._erase_drag = erase
            self._rect_start = yx
            self._update_rect(yx[0], yx[1])

    def _on_release(self, event):
        if not self._painting:
            return
        erase = self._erase_drag
        self._painting = False
        self._erase_drag = False
        if self.tool == "brush":
            return
        yx = self._event_yx(event)
        h = int(self._ct.get_array().shape[0]) if self._ct is not None else 0
        w = int(self._ct.get_array().shape[1]) if self._ct is not None else 0
        if self.tool == "lasso":
            if yx is not None and (
                not self._lasso_pts or yx != self._lasso_pts[-1]
            ):
                self._lasso_pts.append(yx)
            mask = _polygon_mask(h, w, self._lasso_pts) if h and w else None
            self._lasso_pts = []
            if self._lasso_line is not None:
                self._lasso_line.remove()
                self._lasso_line = None
            self.draw_idle()
            if mask is not None and mask.any() and self.region_callback:
                self.region_callback(mask, erase=erase)
            return
        if self.tool == "rect":
            if self._rect_start is None:
                return
            y0, x0 = self._rect_start
            y1, x1 = yx if yx is not None else self._rect_start
            self._rect_start = None
            if self._rect_artist is not None:
                self._rect_artist.remove()
                self._rect_artist = None
            self.draw_idle()
            if h and w and self.region_callback:
                mask = _rect_mask(h, w, y0, x0, y1, x1)
                if mask.any():
                    self.region_callback(mask, erase=erase)

    def _on_motion(self, event):
        yx = self._event_yx(event)
        if yx is None:
            if self.hover_callback:
                self.hover_callback(None, None)
            return
        if self.hover_callback:
            self.hover_callback(yx[0], yx[1])
        if not self._painting:
            return
        if self.tool == "brush":
            if self.paint_callback:
                self.paint_callback(
                    yx[0], yx[1], first=False, erase=self._erase_drag
                )
        elif self.tool == "lasso":
            last = self._lasso_pts[-1] if self._lasso_pts else None
            if last is None or (abs(yx[0] - last[0]) + abs(yx[1] - last[1])) > 1.5:
                self._lasso_pts.append(yx)
                self._update_lasso_line()
        elif self.tool == "rect":
            self._update_rect(yx[0], yx[1])

    def _on_scroll(self, event):
        if self.wheel_callback:
            self.wheel_callback(1 if event.step > 0 else -1)


def _quick_pore_labels(
    ct_z: np.ndarray,
    thresh_closed: int,
    fov2d: np.ndarray,
    thresh_open: int | None = None,
    show_open_hint: bool = True,
    crack_delta: float = 0.0,
    crack_elongation: float = 2.5,
    voxel_um: float = 1.0,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    min_pore_voxels: int = 8,
) -> np.ndarray:
    """Live setup preview — same open/closed/rim rules as the editor (2D envelope)."""
    return pc.preview_labels_for_slice(
        ct_z,
        thresh_closed=thresh_closed,
        fov2d=fov2d,
        thresh_open=thresh_open,
        voxel_um=voxel_um,
        ball_um=ball_um,
        exclude_edge_um=exclude_edge_um,
        crack_delta=crack_delta,
        crack_elongation=crack_elongation,
        min_pore_voxels=min_pore_voxels,
        show_open=show_open_hint,
    )


class _LoadVolumeWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object, object, float)  # vol, folder, pixel_um
    failed = QtCore.pyqtSignal(str)

    def __init__(self, dataset: str, downsample: int):
        super().__init__()
        self.dataset = dataset
        self.downsample = downsample

    def run(self):
        try:
            folder = pc.ROOT / self.dataset
            files = pc.list_tiffs(folder)
            if not files:
                raise FileNotFoundError(f"No TIFF slices in {folder}")
            header = pc.parse_header(folder)
            pixel_um = float(header.get("Pixel Size", 1.0))
            vol = pc.load_volume(files, self.downsample)
            self.finished_ok.emit(vol, folder, pixel_um)
        except Exception:
            self.failed.emit(traceback.format_exc())


class _AnalyzePreviewWorker(QtCore.QThread):
    """Full 3D analyze_volume -> labels (exact match to Initialize)."""

    finished_ok = QtCore.pyqtSignal(object)  # labels ndarray
    failed = QtCore.pyqtSignal(str)

    def __init__(self, vol: np.ndarray, pixel_um: float, downsample: int, kwargs: dict):
        super().__init__()
        self.vol = vol
        self.pixel_um = pixel_um
        self.downsample = downsample
        self.kwargs = kwargs

    def run(self):
        try:
            result = pc.analyze_volume(
                self.vol,
                self.pixel_um,
                self.downsample,
                None,
                **self.kwargs,
            )
            labels = pc.masks_to_labels(
                result["solid"],
                result["closed_pore"],
                result["open_pore"],
                result["open_wrap"],
            )
            self.finished_ok.emit(labels)
        except Exception:
            self.failed.emit(traceback.format_exc())


class _PrepareSessionWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)  # Path
    failed = QtCore.pyqtSignal(str)

    def __init__(self, folder: Path, kwargs: dict):
        super().__init__()
        self.folder = folder
        self.kwargs = kwargs

    def run(self):
        try:
            path = pc.prepare_edit_session(self.folder, **self.kwargs)
            self.finished_ok.emit(path)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SessionSetupWindow(QtWidgets.QMainWindow):
    """Pick dataset + threshold, preview pores, then initialize an edit session."""

    session_ready = QtCore.pyqtSignal(object)  # Path

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Porosity edit — setup")
        self.resize(1180, 820)

        self.vol: np.ndarray | None = None
        self.folder: Path | None = None
        self.pixel_um = 1.0
        self.fov2d: np.ndarray | None = None
        self.otsu_t: int | None = None
        self.air_t: int | None = None
        self.z = 0
        self.ball_um = 15.0
        self.labels_3d: np.ndarray | None = None
        self._load_worker: _LoadVolumeWorker | None = None
        self._prep_worker: _PrepareSessionWorker | None = None
        self._analyze_worker: _AnalyzePreviewWorker | None = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._refresh_preview)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, stretch=3)
        self.canvas = SliceCanvas(self)
        self.canvas.wheel_callback = self._nudge_z
        left.addWidget(NavigationToolbar2QT(self.canvas, self))
        left.addWidget(self.canvas)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, stretch=1)

        right.addWidget(QtWidgets.QLabel("<b>Dataset</b>"))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems(pc.discover_datasets())
        right.addWidget(self.dataset_combo)

        form = QtWidgets.QFormLayout()
        self.ds_spin = QtWidgets.QSpinBox()
        self.ds_spin.setRange(1, 8)
        self.ds_spin.setValue(2)
        form.addRow("Downsample", self.ds_spin)

        self.k_spin = QtWidgets.QDoubleSpinBox()
        self.k_spin.setRange(0.5, 6.0)
        self.k_spin.setSingleStep(0.1)
        self.k_spin.setDecimals(2)
        self.k_spin.setValue(2.5)
        self.k_spin.valueChanged.connect(self._on_k_changed)
        form.addRow("Air k·σ", self.k_spin)
        right.addLayout(form)

        self.btn_load = QtWidgets.QPushButton("Load volume for preview")
        self.btn_load.clicked.connect(self.load_preview_volume)
        right.addWidget(self.btn_load)

        right.addWidget(QtWidgets.QLabel("<b>Vesicle / circle pore mix (blue)</b>"))
        mix_row = QtWidgets.QHBoxLayout()
        self.mix_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(50)
        self.mix_slider.valueChanged.connect(self._on_mix_changed)
        mix_row.addWidget(self.mix_slider)
        self.mix_value = QtWidgets.QLabel("0.50")
        mix_row.addWidget(self.mix_value)
        right.addLayout(mix_row)

        right.addWidget(QtWidgets.QLabel("<b>Open-pore mix (yellow)</b>"))
        open_row = QtWidgets.QHBoxLayout()
        self.open_mix_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.open_mix_slider.setRange(0, 100)
        self.open_mix_slider.setValue(50)
        self.open_mix_slider.valueChanged.connect(self._on_open_mix_changed)
        open_row.addWidget(self.open_mix_slider)
        self.open_mix_value = QtWidgets.QLabel("0.50")
        open_row.addWidget(self.open_mix_value)
        right.addLayout(open_row)

        self.chk_lock_open = QtWidgets.QCheckBox("Lock open mix to vesicle mix")
        self.chk_lock_open.setChecked(True)
        self.chk_lock_open.stateChanged.connect(self._on_lock_open)
        right.addWidget(self.chk_lock_open)

        right.addWidget(QtWidgets.QLabel("<b>Crack pores (separate)</b>"))
        self.chk_cracks = QtWidgets.QCheckBox("Enable crack detection")
        self.chk_cracks.setChecked(False)
        self.chk_cracks.stateChanged.connect(self._on_crack_toggled)
        right.addWidget(self.chk_cracks)

        crack_form = QtWidgets.QFormLayout()
        self.crack_delta_spin = QtWidgets.QDoubleSpinBox()
        self.crack_delta_spin.setRange(200.0, 3000.0)
        self.crack_delta_spin.setSingleStep(50.0)
        self.crack_delta_spin.setDecimals(0)
        self.crack_delta_spin.setValue(1100.0)
        self.crack_delta_spin.setEnabled(False)
        self.crack_delta_spin.valueChanged.connect(self._on_params_changed)
        crack_form.addRow("Crack contrast Δ", self.crack_delta_spin)

        self.crack_elong_spin = QtWidgets.QDoubleSpinBox()
        self.crack_elong_spin.setRange(1.0, 8.0)
        self.crack_elong_spin.setSingleStep(0.25)
        self.crack_elong_spin.setDecimals(2)
        self.crack_elong_spin.setValue(2.50)
        self.crack_elong_spin.setEnabled(False)
        self.crack_elong_spin.valueChanged.connect(self._on_params_changed)
        crack_form.addRow("Min elongation", self.crack_elong_spin)
        right.addLayout(crack_form)

        hint = QtWidgets.QLabel(
            "Live 2D preview and 3D Initialize use the <b>same blue closed-pore "
            "rule</b> (per-slice fill-holes at vesicle T).<br>"
            "Accurate 3D adds rolling-ball open/yellow + cupping extras.<br>"
            "0 = air T, 1 = Otsu."
        )
        hint.setWordWrap(True)
        right.addWidget(hint)

        self.chk_open_hint = QtWidgets.QCheckBox("Show open pores (yellow)")
        self.chk_open_hint.setChecked(True)
        self.chk_open_hint.stateChanged.connect(self._on_params_changed)
        right.addWidget(self.chk_open_hint)

        self.t_info = QtWidgets.QLabel("Load a volume to see air / Otsu / mixed T.")
        self.t_info.setWordWrap(True)
        right.addWidget(self.t_info)

        right.addWidget(QtWidgets.QLabel("<b>Preview slice Z</b>"))
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.setEnabled(False)
        self.z_slider.valueChanged.connect(self._set_z)
        right.addWidget(self.z_slider)
        self.z_label = QtWidgets.QLabel("—")
        right.addWidget(self.z_label)

        self.preview_mode = QtWidgets.QLabel("Preview: —")
        self.preview_mode.setWordWrap(True)
        right.addWidget(self.preview_mode)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        right.addStretch(1)

        self.btn_3d = QtWidgets.QPushButton("Accurate 3D preview")
        self.btn_3d.setEnabled(False)
        self.btn_3d.clicked.connect(self.run_accurate_3d_preview)
        right.addWidget(self.btn_3d)

        self.btn_init = QtWidgets.QPushButton("Initialize edit session")
        self.btn_init.setStyleSheet(
            "QPushButton { background:#0D47A1; color:white; font-weight:700; padding:10px; }"
        )
        self.btn_init.clicked.connect(self.initialize_session)
        self.btn_init.setEnabled(False)
        right.addWidget(self.btn_init)

        self.btn_open = QtWidgets.QPushButton("Open existing session…")
        self.btn_open.clicked.connect(self.open_existing_session)
        right.addWidget(self.btn_open)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        right.addWidget(self.progress)

        self.statusBar().showMessage(
            "Choose a dataset, load preview, tune mix, then Accurate 3D preview / Initialize."
        )

    def _mix(self) -> float:
        return self.mix_slider.value() / 100.0

    def _open_mix(self) -> float:
        return self.open_mix_slider.value() / 100.0

    def _mixed_t(self) -> int | None:
        if self.air_t is None or self.otsu_t is None:
            return None
        return int(round(self.air_t + self._mix() * (self.otsu_t - self.air_t)))

    def _open_t(self) -> int | None:
        if self.air_t is None or self.otsu_t is None:
            return None
        return int(round(self.air_t + self._open_mix() * (self.otsu_t - self.air_t)))

    def _on_lock_open(self, _state: int = 0):
        locked = self.chk_lock_open.isChecked()
        self.open_mix_slider.setEnabled(not locked and self.vol is not None)
        if locked:
            self.open_mix_slider.blockSignals(True)
            self.open_mix_slider.setValue(self.mix_slider.value())
            self.open_mix_slider.blockSignals(False)
            self.open_mix_value.setText(f"{self._open_mix():.2f}")
            self._update_t_info()
        self._on_params_changed()

    def _on_crack_toggled(self, _state: int = 0):
        on = self.chk_cracks.isChecked() and self.vol is not None
        self.crack_delta_spin.setEnabled(on)
        self.crack_elong_spin.setEnabled(on)
        self._on_params_changed()

    def _crack_delta(self) -> float:
        if not self.chk_cracks.isChecked():
            return 0.0
        return float(self.crack_delta_spin.value())

    def _invalidate_3d(self):
        self.labels_3d = None

    def _on_params_changed(self, *_args):
        self._invalidate_3d()
        self._update_t_info()
        self._preview_timer.start()

    def _on_mix_changed(self, _v: int = 0):
        self.mix_value.setText(f"{self._mix():.2f}")
        if self.chk_lock_open.isChecked():
            self.open_mix_slider.blockSignals(True)
            self.open_mix_slider.setValue(self.mix_slider.value())
            self.open_mix_slider.blockSignals(False)
            self.open_mix_value.setText(f"{self._open_mix():.2f}")
        self._on_params_changed()

    def _on_open_mix_changed(self, _v: int = 0):
        if self.chk_lock_open.isChecked():
            return
        self.open_mix_value.setText(f"{self._open_mix():.2f}")
        self._on_params_changed()

    def _on_k_changed(self, _v: float = 0.0):
        if self.vol is None or self.fov2d is None or self.otsu_t is None:
            return
        air_mask = pc.exterior_air_mask(self.vol, self.fov2d, self.otsu_t)
        try:
            self.air_t = pc.pick_air_threshold(
                self.vol, air_mask, k_std=float(self.k_spin.value())
            )
        except RuntimeError:
            self.air_t = self.otsu_t
        self._on_params_changed()

    def _update_t_info(self):
        if self.air_t is None or self.otsu_t is None:
            return
        tc = self._mixed_t()
        to = self._open_t()
        self.t_info.setText(
            f"Air T = {self.air_t}<br>"
            f"Otsu T = {self.otsu_t}<br>"
            f"<b>Vesicle T = {tc}</b> (mix={self._mix():.2f})<br>"
            f"<b>Open T = {to}</b> (mix={self._open_mix():.2f})<br>"
            f"Crack Δ = {self._crack_delta():.0f} "
            f"(elong≥{self.crack_elong_spin.value():.2f})"
        )

    def _voxel_um(self) -> float:
        return float(self.pixel_um) * float(self.ds_spin.value())

    def _seg_kwargs(self) -> dict:
        """Shared kwargs for Accurate 3D preview and Initialize."""
        return dict(
            air_k_std=float(self.k_spin.value()),
            thresh_mix=self._mix(),
            open_thresh_mix=self._open_mix(),
            crack_delta=self._crack_delta(),
            crack_elongation=float(self.crack_elong_spin.value()),
            crack_um=12.0,
            ball_um=self.ball_um,
            cupping_um=40.0,
            cupping_delta=1000.0,
        )

    def _nudge_z(self, delta: int):
        if self.vol is None:
            return
        self.z_slider.setValue(int(np.clip(self.z + delta, 0, self.vol.shape[0] - 1)))

    def _set_z(self, z: int):
        self.z = int(z)
        self.z_label.setText(f"Z = {self.z}")
        self._preview_timer.start()

    def _set_busy(self, busy: bool, msg: str = ""):
        self.progress.setVisible(busy)
        self.btn_load.setEnabled(not busy)
        self.btn_init.setEnabled(not busy and self.vol is not None)
        self.btn_3d.setEnabled(not busy and self.vol is not None)
        self.btn_open.setEnabled(not busy)
        self.dataset_combo.setEnabled(not busy)
        self.ds_spin.setEnabled(not busy)
        self.k_spin.setEnabled(not busy)
        self.mix_slider.setEnabled(not busy and self.vol is not None)
        self.open_mix_slider.setEnabled(
            not busy and self.vol is not None and not self.chk_lock_open.isChecked()
        )
        self.chk_lock_open.setEnabled(not busy and self.vol is not None)
        self.chk_cracks.setEnabled(not busy and self.vol is not None)
        crack_on = (not busy) and self.vol is not None and self.chk_cracks.isChecked()
        self.crack_delta_spin.setEnabled(crack_on)
        self.crack_elong_spin.setEnabled(crack_on)
        if msg:
            self.status.setText(msg)
            self.statusBar().showMessage(msg)

    def load_preview_volume(self):
        if self._load_worker and self._load_worker.isRunning():
            return
        name = self.dataset_combo.currentText()
        ds = int(self.ds_spin.value())
        self._invalidate_3d()
        self._set_busy(True, f"Loading {name} (downsample={ds})…")
        self._load_worker = _LoadVolumeWorker(name, ds)
        self._load_worker.finished_ok.connect(self._on_volume_loaded)
        self._load_worker.failed.connect(self._on_volume_failed)
        self._load_worker.start()

    def _on_volume_failed(self, err: str):
        self._set_busy(False, "Load failed.")
        QtWidgets.QMessageBox.critical(self, "Load failed", err)

    def _on_volume_loaded(self, vol: np.ndarray, folder: Path, pixel_um: float):
        self.vol = vol
        self.folder = folder
        self.pixel_um = pixel_um
        self.fov2d = pc.fov_disk(vol.shape[1:])
        self.otsu_t = pc.pick_otsu(vol, self.fov2d)
        air_mask = pc.exterior_air_mask(vol, self.fov2d, self.otsu_t)
        try:
            self.air_t = pc.pick_air_threshold(vol, air_mask, k_std=float(self.k_spin.value()))
        except RuntimeError:
            self.air_t = self.otsu_t
        self.z = vol.shape[0] // 2
        self.z_slider.blockSignals(True)
        self.z_slider.setEnabled(True)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(vol.shape[0] - 1)
        self.z_slider.setValue(self.z)
        self.z_slider.blockSignals(False)
        self.z_label.setText(f"Z = {self.z}")
        self._update_t_info()
        self._set_busy(False, f"Loaded {folder.name}  shape={vol.shape}")
        self.btn_init.setEnabled(True)
        self.btn_3d.setEnabled(True)
        self.mix_slider.setEnabled(True)
        self.open_mix_slider.setEnabled(not self.chk_lock_open.isChecked())
        self.chk_lock_open.setEnabled(True)
        self.chk_cracks.setEnabled(True)
        self._on_crack_toggled()
        self._refresh_preview()

    def _refresh_preview(self):
        if self.vol is None or self.fov2d is None:
            return
        t = self._mixed_t()
        t_open = self._open_t()
        if t is None or t_open is None:
            return
        ct = self.vol[self.z]
        if self.labels_3d is not None:
            labels = self.labels_3d[self.z]
            mode = "3D accurate (matches Initialize)"
        else:
            labels = _quick_pore_labels(
                ct,
                t,
                self.fov2d,
                thresh_open=t_open,
                show_open_hint=self.chk_open_hint.isChecked(),
                crack_delta=self._crack_delta(),
                crack_elongation=float(self.crack_elong_spin.value()),
                voxel_um=self._voxel_um(),
                ball_um=self.ball_um,
                exclude_edge_um=self.ball_um,
            )
            mode = "2D approx (same rules; use Accurate 3D for exact match)"
        self.preview_mode.setText(f"Preview: {mode}")
        crack_txt = (
            f" crackΔ={self._crack_delta():.0f}"
            if self.chk_cracks.isChecked()
            else " cracks=off"
        )
        self.canvas.show_slice(
            ct,
            labels,
            f"{self.folder.name if self.folder else ''}  Z={self.z}  "
            f"Tv={t} To={t_open}{crack_txt}",
            opacity=0.75,
        )

    def run_accurate_3d_preview(self):
        if self.vol is None:
            QtWidgets.QMessageBox.warning(self, "No volume", "Load a preview volume first.")
            return
        if self._analyze_worker and self._analyze_worker.isRunning():
            return
        kwargs = self._seg_kwargs()
        self._set_busy(
            True,
            "Running full 3D segmentation preview (rolling-ball + cupping)…\n"
            "This can take a few minutes.",
        )
        self._analyze_worker = _AnalyzePreviewWorker(
            self.vol,
            self.pixel_um,
            int(self.ds_spin.value()),
            kwargs,
        )
        self._analyze_worker.finished_ok.connect(self._on_3d_preview_ok)
        self._analyze_worker.failed.connect(self._on_3d_preview_failed)
        self._analyze_worker.start()

    def _on_3d_preview_failed(self, err: str):
        self._set_busy(False, "3D preview failed.")
        self.btn_3d.setEnabled(self.vol is not None)
        self.btn_init.setEnabled(self.vol is not None)
        QtWidgets.QMessageBox.critical(self, "3D preview failed", err)

    def _on_3d_preview_ok(self, labels: np.ndarray):
        self.labels_3d = labels
        self._set_busy(False, "3D accurate preview ready — matches Initialize.")
        self.btn_3d.setEnabled(True)
        self.btn_init.setEnabled(True)
        self._refresh_preview()

    def initialize_session(self):
        if self.folder is None:
            QtWidgets.QMessageBox.warning(self, "No volume", "Load a preview volume first.")
            return
        if self._prep_worker and self._prep_worker.isRunning():
            return

        session = pc.session_dir(self.folder.name)
        if (session / "labels.npy").exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite session?",
                f"{session} already exists.\n\n"
                "Initialize will re-run auto-segmentation and overwrite labels "
                "(manual edits in that session will be lost).\n\nContinue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        kwargs = dict(
            downsample=int(self.ds_spin.value()),
            **self._seg_kwargs(),
        )
        self._set_busy(
            True,
            "Initializing full 3D edit session (rolling-ball + cupping)…\n"
            "This can take a few minutes.",
        )
        self._prep_worker = _PrepareSessionWorker(self.folder, kwargs)
        self._prep_worker.finished_ok.connect(self._on_prepare_ok)
        self._prep_worker.failed.connect(self._on_prepare_failed)
        self._prep_worker.start()

    def _on_prepare_failed(self, err: str):
        self._set_busy(False, "Initialize failed.")
        self.btn_init.setEnabled(self.vol is not None)
        QtWidgets.QMessageBox.critical(self, "Initialize failed", err)

    def _on_prepare_ok(self, session: Path):
        self._set_busy(False, f"Session ready: {session}")
        self.btn_init.setEnabled(True)
        self.session_ready.emit(session)

    def open_existing_session(self):
        start = str(pc.ROOT / "_edit_sessions")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open edit session folder", start
        )
        if not path:
            return
        session = Path(path)
        if not (session / "labels.npy").exists() or not (session / "volume.npy").exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete session",
                f"{session} is missing labels.npy or volume.npy",
            )
            return
        self.session_ready.emit(session)


class PorosityEditor(QtWidgets.QMainWindow):
    def __init__(self, session: Path):
        super().__init__()
        self.session = Path(session)
        self.vol = np.load(self.session / "volume.npy")
        self.labels = np.load(self.session / "labels.npy")
        self.meta = json.loads((self.session / "meta.json").read_text(encoding="utf-8"))
        # Remap absolute source_folder from another machine if needed
        name = self.meta.get("dataset", self.session.name)
        src = pc.resolve_project_path(self.meta.get("source_folder"), fallback_name=name)
        if src is not None:
            rel = pc.path_relative_to_root(src)
            if self.meta.get("source_folder") != rel:
                self.meta["source_folder"] = rel
                (self.session / "meta.json").write_text(
                    json.dumps(self.meta, indent=2), encoding="utf-8"
                )
        # Resume at last saved slice if present
        self.z = int(self.meta.get("last_z", self.vol.shape[0] // 2))
        self.z = int(np.clip(self.z, 0, self.vol.shape[0] - 1))
        self.brush = 4
        self.label = pc.L_CLOSED
        self.tool = "brush"  # brush | lasso | rect
        self.opacity = 0.7
        self.dirty = False
        self._undo: list[tuple[int, np.ndarray]] = []
        self._hover = (None, None)
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(120_000)  # 2 min
        self._autosave_timer.timeout.connect(self._autosave_if_dirty)

        name = self.meta.get("dataset", self.session.name)
        self.setWindowTitle(f"Porosity label editor — {name}")
        self.resize(1100, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, stretch=3)
        self.canvas = SliceCanvas(self)
        self.canvas.paint_callback = self._paint_at
        self.canvas.region_callback = self._apply_region
        self.canvas.hover_callback = self._on_hover
        self.canvas.wheel_callback = self._nudge_z
        left.addWidget(NavigationToolbar2QT(self.canvas, self))
        left.addWidget(self.canvas)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, stretch=1)

        right.addWidget(QtWidgets.QLabel("<b>Slice Z</b>"))
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(self.vol.shape[0] - 1)
        self.z_slider.setValue(self.z)
        self.z_slider.valueChanged.connect(self._set_z)
        right.addWidget(self.z_slider)
        self.z_label = QtWidgets.QLabel()
        right.addWidget(self.z_label)

        right.addWidget(QtWidgets.QLabel("<b>Tool</b>"))
        self.tool_group = QtWidgets.QButtonGroup(self)
        for tid, text in (
            ("brush", "Brush"),
            ("lasso", "Lasso select"),
            ("rect", "Box select"),
        ):
            btn = QtWidgets.QRadioButton(text)
            if tid == self.tool:
                btn.setChecked(True)
            self.tool_group.addButton(btn)
            self.tool_group.setId(btn, {"brush": 0, "lasso": 1, "rect": 2}[tid])
            right.addWidget(btn)
        self.tool_group.idClicked.connect(self._set_tool_id)

        right.addWidget(QtWidgets.QLabel("<b>Paint label</b>"))
        self.label_group = QtWidgets.QButtonGroup(self)
        for code, text, color in LABEL_BUTTONS:
            btn = QtWidgets.QRadioButton(text)
            btn.setStyleSheet(f"QRadioButton {{ color: {color}; font-weight: 600; }}")
            if code == self.label:
                btn.setChecked(True)
            self.label_group.addButton(btn, code)
            right.addWidget(btn)
        self.label_group.idClicked.connect(self._set_label)

        right.addWidget(QtWidgets.QLabel("<b>Brush radius (px)</b>"))
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(40)
        self.brush_slider.setValue(self.brush)
        self.brush_slider.valueChanged.connect(self._set_brush)
        right.addWidget(self.brush_slider)
        self.brush_label = QtWidgets.QLabel(f"{self.brush} px")
        right.addWidget(self.brush_label)

        right.addWidget(QtWidgets.QLabel("<b>Overlay strength</b>"))
        self.op_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.op_slider.setMinimum(10)
        self.op_slider.setMaximum(100)
        self.op_slider.setValue(int(self.opacity * 100))
        self.op_slider.valueChanged.connect(self._set_opacity)
        right.addWidget(self.op_slider)

        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        right.addWidget(self.stats)

        help_txt = QtWidgets.QLabel(
            "<b>Display:</b> blue / yellow pore shadows<br><br>"
            "<b>Tools:</b> Brush | Lasso | Box select<br>"
            "Left-drag: paint selected label<br>"
            "Right-drag: erase → solid<br>"
            "Lasso/Box: drag a region, release to fill<br><br>"
            "Mouse wheel: Z &nbsp;|&nbsp; Keys 0–4: label<br>"
            "B / L / X: tool &nbsp;|&nbsp; [ ]: brush size<br>"
            "Ctrl+S: save &nbsp;|&nbsp; Ctrl+Z: undo"
        )
        help_txt.setWordWrap(True)
        right.addWidget(help_txt)
        right.addStretch(1)

        self.btn_save = QtWidgets.QPushButton("Save (keep editing later)")
        self.btn_save.clicked.connect(lambda: self.save_labels(quit=False))
        right.addWidget(self.btn_save)

        self.btn_save_quit = QtWidgets.QPushButton("Save & quit")
        self.btn_save_quit.clicked.connect(self.save_and_quit)
        right.addWidget(self.btn_save_quit)

        self.btn_reset = QtWidgets.QPushButton("Reset slice to auto")
        self.btn_reset.clicked.connect(self.reset_slice_to_auto)
        right.addWidget(self.btn_reset)

        self.btn_confirm = QtWidgets.QPushButton("Confirm & compute porosity")
        self.btn_confirm.setStyleSheet(
            "QPushButton { background:#1B5E20; color:white; font-weight:700; padding:8px; }"
        )
        self.btn_confirm.clicked.connect(self.confirm_and_compute)
        right.addWidget(self.btn_confirm)

        self._refresh()
        saved = self.meta.get("last_saved_unix")
        if saved:
            self.statusBar().showMessage(
                f"Resumed {self.session}  |  last save {time.strftime('%Y-%m-%d %H:%M', time.localtime(saved))}"
            )
        else:
            self.statusBar().showMessage(f"{self.session}  |  Ctrl+S to save and continue later")
        self._autosave_timer.start()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        mods = event.modifiers()
        if mods == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_S:
            self.save_labels(quit=False)
            return
        if mods == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z:
            self.undo()
            return
        if QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_4:
            code = key - QtCore.Qt.Key_0
            btn = self.label_group.button(code)
            if btn:
                btn.setChecked(True)
            self._set_label(code)
            return
        if key == QtCore.Qt.Key_BracketLeft:
            self.brush_slider.setValue(max(1, self.brush - 1))
            return
        if key == QtCore.Qt.Key_BracketRight:
            self.brush_slider.setValue(min(40, self.brush + 1))
            return
        if key == QtCore.Qt.Key_B:
            self._set_tool("brush")
            return
        if key == QtCore.Qt.Key_L:
            self._set_tool("lasso")
            return
        if key == QtCore.Qt.Key_X:
            self._set_tool("rect")
            return
        super().keyPressEvent(event)

    def _set_z(self, z: int):
        self.z = int(z)
        self._refresh()

    def _nudge_z(self, dz: int):
        self.z_slider.setValue(int(np.clip(self.z + dz, 0, self.vol.shape[0] - 1)))

    def _set_label(self, code: int):
        self.label = int(code)

    def _set_tool_id(self, tid: int):
        self._set_tool({0: "brush", 1: "lasso", 2: "rect"}.get(tid, "brush"))

    def _set_tool(self, tool: str):
        self.tool = tool
        self.canvas.tool = tool
        self.canvas.clear_tool_artists()
        btn = self.tool_group.button({"brush": 0, "lasso": 1, "rect": 2}[tool])
        if btn and not btn.isChecked():
            btn.setChecked(True)
        self.brush_slider.setEnabled(tool == "brush")
        y, x = self._hover
        self.canvas.set_brush_cursor(y, x, self.brush)
        self.statusBar().showMessage(
            {
                "brush": "Brush: left=paint, right=erase→solid",
                "lasso": "Lasso: drag around a region, release to fill (right=erase)",
                "rect": "Box: drag a rectangle, release to fill (right=erase)",
            }[tool],
            5000,
        )

    def _set_brush(self, r: int):
        self.brush = int(r)
        self.brush_label.setText(f"{self.brush} px")
        y, x = self._hover
        self.canvas.set_brush_cursor(y, x, self.brush)

    def _set_opacity(self, pct: int):
        self.opacity = pct / 100.0
        self._refresh()

    def _on_hover(self, y, x):
        self._hover = (y, x)
        self.canvas.set_brush_cursor(y, x, self.brush)

    def _push_undo(self):
        self._undo.append((self.z, self.labels[self.z].copy()))
        if len(self._undo) > 40:
            self._undo.pop(0)

    def undo(self):
        if not self._undo:
            self.statusBar().showMessage("Nothing to undo")
            return
        z, sl = self._undo.pop()
        self.labels[z] = sl
        self.dirty = True
        if z == self.z:
            self._refresh()
        else:
            self.z_slider.setValue(z)
        self.statusBar().showMessage(f"Undo slice z={z}")

    def _paint_code(self, erase: bool) -> int:
        return pc.L_SOLID if erase else self.label

    def _paint_at(self, y: float, x: float, first: bool = False, erase: bool = False):
        if first:
            self._push_undo()
        h, w = self.labels.shape[1:]
        mask = _disk_mask(h, w, y, x, self.brush)
        self.labels[self.z][mask] = self._paint_code(erase)
        self.dirty = True
        self._refresh(stats_only=False)

    def _apply_region(self, mask: np.ndarray, erase: bool = False):
        self._push_undo()
        self.labels[self.z][mask] = self._paint_code(erase)
        self.dirty = True
        n = int(mask.sum())
        action = "Erased" if erase else f"Painted {pc.LABEL_NAMES.get(self.label, self.label)}"
        self.statusBar().showMessage(f"{action} {n:,} voxels", 4000)
        self._refresh(stats_only=False)

    def _live_phi(self) -> str:
        solid = int((self.labels == pc.L_SOLID).sum())
        closed = int((self.labels == pc.L_CLOSED).sum())
        opened = int((self.labels == pc.L_OPEN).sum())
        wrap = int((self.labels == pc.L_WRAP).sum())
        den = solid + closed + opened
        if den == 0:
            return "No particle voxels yet."
        dirty = "unsaved" if self.dirty else "saved"
        return (
            f"<b>Live porosity ({dirty})</b><br>"
            f"closed={100 * closed / den:.2f}%  (blue shadow)<br>"
            f"open={100 * opened / den:.2f}%  (yellow shadow)<br>"
            f"total={100 * (closed + opened) / den:.2f}%<br>"
            f"wrap/ignore={wrap:,} (not drawn)<br>"
            f"auto total was {self.meta.get('phi_total_auto_pct', float('nan')):.2f}%"
        )

    def _refresh(self, stats_only: bool = False):
        self.z_label.setText(f"z = {self.z} / {self.vol.shape[0] - 1}")
        self.stats.setText(self._live_phi())
        if stats_only:
            return
        title = (
            f"{self.meta.get('dataset', '')}  z={self.z}  "
            f"tool={self.tool}  paint={pc.LABEL_NAMES.get(self.label, self.label)}"
            + ("  *" if self.dirty else "")
        )
        self.canvas.show_slice(self.vol[self.z], self.labels[self.z], title, self.opacity)
        y, x = self._hover
        self.canvas.set_brush_cursor(y, x, self.brush)

    def _write_meta(self):
        self.meta["last_z"] = int(self.z)
        self.meta["last_saved_unix"] = time.time()
        self.meta["confirmed"] = bool(self.meta.get("confirmed", False))
        (self.session / "meta.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

    def save_labels(self, quit: bool = False) -> bool:
        np.save(self.session / "labels.npy", self.labels)
        self._write_meta()
        self.dirty = False
        self._refresh()
        when = time.strftime("%H:%M:%S")
        self.statusBar().showMessage(
            f"Saved {self.session / 'labels.npy'} at {when} — reopen this session to continue",
            8000,
        )
        return True

    def _autosave_if_dirty(self):
        if self.dirty:
            self.save_labels(quit=False)
            self.statusBar().showMessage("Autosaved", 4000)

    def save_and_quit(self):
        self.save_labels(quit=False)
        self.close()

    def reset_slice_to_auto(self):
        auto_path = self.session / "labels_auto.npy"
        if not auto_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "labels_auto.npy not found.")
            return
        auto = np.load(auto_path, mmap_mode="r")
        self._push_undo()
        self.labels[self.z] = np.array(auto[self.z])
        self.dirty = True
        self._refresh()

    def confirm_and_compute(self):
        if self.dirty:
            self.save_labels(quit=False)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm edited labels?",
            "Compute porosity and QC figures from the current labels?\n"
            "This uses your edits as ground truth (no re-thresholding).\n\n"
            "You can still reopen the session later and edit again.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.statusBar().showMessage("Computing porosity…")
        QtWidgets.QApplication.processEvents()
        try:
            row = pc.finalize_edit_session(self.session, out_dir=pc.ROOT / "_porosity_results")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Finalize failed", str(e))
            return
        QtWidgets.QMessageBox.information(
            self,
            "Done",
            "Edited porosity\n"
            f"closed={row['phi_closed_pct']:.2f}%\n"
            f"open={row['phi_open_pct']:.2f}%\n"
            f"total={row['phi_total_pct']:.2f}%\n\n"
            f"QC: {row['qc_dir']}",
        )
        self.statusBar().showMessage("Finalize complete", 8000)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "Save labels before closing?\n"
                "Saved work can be resumed later with the same --session path.",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_labels(quit=False)
        self._autosave_timer.stop()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual CT pore-label editor.")
    parser.add_argument(
        "--session",
        default=None,
        help='Optional path like "_edit_sessions/6-11" (skip setup and open editor)',
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    editor_holder: dict[str, PorosityEditor | None] = {"win": None}

    def open_editor(session: Path):
        session = Path(session)
        if not session.is_absolute():
            session = pc.ROOT / session
        if not (session / "labels.npy").exists() or not (session / "volume.npy").exists():
            QtWidgets.QMessageBox.critical(
                None,
                "Session incomplete",
                f"{session}\nis missing labels.npy or volume.npy",
            )
            return
        if editor_holder["win"] is not None:
            editor_holder["win"].close()
        win = PorosityEditor(session)
        editor_holder["win"] = win
        win.show()
        win.raise_()
        win.activateWindow()

    if args.session:
        open_editor(Path(args.session))
    else:
        setup = SessionSetupWindow()
        setup.session_ready.connect(lambda p: (setup.hide(), open_editor(p)))
        setup.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
