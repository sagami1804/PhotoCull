"""失敗写真チェッカー: Windows デスクトップ GUI (PySide6)"""
from __future__ import annotations

import csv
import faulthandler
import os
import shutil
import statistics
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QAction, QColor, QFont, QIcon, QImage, QKeySequence,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSplitter,
                               QTextBrowser, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

from . import analyzer as az
from . import people

APP_NAME = "失敗写真チェッカー"
ROLE_PATH = Qt.ItemDataRole.UserRole
ICON_W, ICON_H = az.THUMB_SIZE, az.THUMB_SIZE


LOG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PhotoCull"
LOG_FILE = LOG_DIR / "error.log"


def log_exception(where: str):
    """予期しないエラーをログファイルに残す（exe 版はコンソールが無いため）。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {where}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass


def install_crash_handlers():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Python 外 (OpenCV / Qt 内部) で落ちた場合もスタックを記録する
    global _fault_file
    _fault_file = open(LOG_DIR / "crash.log", "a", encoding="utf-8")
    faulthandler.enable(_fault_file)

    def hook(exc_type, exc, tb):
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 未処理の例外\n")
                f.write("".join(traceback.format_exception(exc_type, exc, tb)))
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None and threading.current_thread() is threading.main_thread():
            QMessageBox.warning(None, APP_NAME,
                                f"エラーが発生しましたが、作業は続けられます。\n\n{exc_type.__name__}: {exc}\n\n"
                                f"詳細: {LOG_FILE}")

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def normalize_query(text: str) -> str:
    """検索用: 小文字にし、カタカナをひらがなにそろえる（「さーふ」でも「サーフ」でも見つかるように）"""
    out = []
    for ch in text.lower().strip():
        o = ord(ch)
        out.append(chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch)
    return "".join(out)


def tag_matches(cls: int, query: str) -> bool:
    if not query:
        return True
    q = normalize_query(query)
    return q in normalize_query(people.LABELS_JA[cls]) or q in people.LABELS_EN[cls]


def pil_to_qimage(img) -> QImage:
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    q = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return q.copy()  # data の寿命から切り離す


# ---------------------------------------------------------------- 解析スレッド
class AnalyzeWorker(QThread):
    result = Signal(object, object)    # Metrics, QImage | None
    progress = Signal(int, int)

    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._run()
        except Exception:
            log_exception("解析スレッド")

    def _run(self):
        cv2.setNumThreads(1)
        total = len(self.paths)
        done = 0
        # 画像の読み込みと顔検出は CPU で並列に行う。巨大な画像を同時に開きすぎると
        # メモリ不足になるので最大 8 枚まで（1024px で読むようになり、8 並列でも十分に軽い）
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._one, p): p for p in self.paths}
            for f in as_completed(futs):
                if self._stop.is_set():
                    for other in futs:
                        other.cancel()
                    break
                try:
                    m, qimg = f.result()
                except Exception as e:  # 念のため: 1枚の失敗で全体を止めない
                    log_exception(f"解析: {futs[f]}")
                    m, qimg = az.Metrics(path=futs[f], ok=False, error=f"解析できません: {e}"), None
                done += 1
                self.result.emit(m, qimg)
                self.progress.emit(done, total)

    def _one(self, path):
        if self._stop.is_set():
            return az.Metrics(path=path, ok=False, error="中止"), None
        m, thumb = az.analyze(path)
        qimg = None
        if thumb is not None:
            try:
                qimg = pil_to_qimage(thumb)
            except Exception:
                qimg = None
        return m, qimg


# ---------------------------------------------------------------- 書き出し先の選択
class ExportDialog(QDialog):
    """移動/コピー先を選ぶ画面。元フォルダの中に作るか、好きなフォルダを指定する。"""

    def __init__(self, parent, kind: str, move: bool, count: int, settings: QSettings):
        super().__init__(parent)
        self.settings = settings
        self.kind = kind
        label = "失敗写真" if kind == "fail" else "OK の写真"
        verb = "移動" if move else "コピー"
        sub = az.OUTPUT_DIR_NAME if kind == "fail" else az.OK_DIR_NAME
        self.setWindowTitle(f"{label}を{verb}")
        self.setMinimumWidth(520)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"<b>{label} {count} 枚</b>を{verb}します。{verb}先を選んでください。"))

        self.rb_sub = QRadioButton(f"元の写真と同じフォルダの中に「{sub}」フォルダを作って入れる")
        self.rb_dir = QRadioButton("フォルダを指定する（新しいフォルダも作れます）")
        v.addWidget(self.rb_sub)
        v.addWidget(self.rb_dir)

        row = QHBoxLayout()
        row.addSpacing(24)
        self.ed_dir = QLineEdit(settings.value(f"export/{kind}_dir", ""))
        self.ed_dir.setPlaceholderText("フォルダを選ぶか、作りたいフォルダの場所を入力")
        browse = QPushButton("参照…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.ed_dir, 1)
        row.addWidget(browse)
        v.addLayout(row)
        hint = QLabel("「参照…」の画面にある「新しいフォルダー」ボタンで、その場でフォルダを作れます。\n"
                      "入力したフォルダが無いときは自動で作ります。同じ名前の写真があるときは上書きせず番号を付けます。")
        hint.setStyleSheet("color:#666; margin-left:24px;")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.cb_raw = QCheckBox(f"同じ名前の RAW ファイル（.RAF / .NEF / .CR3 / .ARW など）も一緒に{verb}する")
        self.cb_raw.setChecked(settings.value("export/with_raw", True) in (True, "true", "1", 1))
        v.addWidget(self.cb_raw)
        if move:
            note = QLabel("移動したあとは「↶ 移動を元に戻す」(Ctrl+Z) で戻せます。")
            note.setStyleSheet("color:#666;")
            v.addWidget(note)

        use_dir = settings.value(f"export/{kind}_mode", "sub") == "dir"
        (self.rb_dir if use_dir else self.rb_sub).setChecked(True)
        self.rb_dir.toggled.connect(self._update)
        self.ed_dir.textChanged.connect(self._update)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(f"{verb}する")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("やめる")
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self._update()

    def _browse(self):
        start = self.ed_dir.text() or self.settings.value("lastdir", "")
        d = QFileDialog.getExistingDirectory(self, "移動先・コピー先のフォルダを選択", start)
        if d:
            self.ed_dir.setText(os.path.normpath(d))
            self.rb_dir.setChecked(True)

    def _update(self):
        ok = self.rb_sub.isChecked() or bool(self.ed_dir.text().strip())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def _accept(self):
        if self.rb_dir.isChecked():
            d = self.ed_dir.text().strip().strip('"')
            if not os.path.isabs(d):
                QMessageBox.warning(self, APP_NAME, "フォルダは C:\\ などから始まる場所で指定してください。")
                return
            if os.path.exists(d) and not os.path.isdir(d):
                QMessageBox.warning(self, APP_NAME, "指定した場所はフォルダではありません。")
                return
            self.settings.setValue(f"export/{self.kind}_dir", d)
        self.settings.setValue(f"export/{self.kind}_mode", "dir" if self.rb_dir.isChecked() else "sub")
        self.settings.setValue("export/with_raw", self.cb_raw.isChecked())
        self.accept()

    def with_raw(self) -> bool:
        return self.cb_raw.isChecked()

    def destination(self) -> str | None:
        return os.path.normpath(self.ed_dir.text().strip().strip('"')) if self.rb_dir.isChecked() else None


# ---------------------------------------------------------------- メイン画面
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.settings = QSettings("PhotoCull", "PhotoCull")
        self.th = self._load_thresholds()

        self.metrics: dict[str, az.Metrics] = {}
        self.thumbs: dict[str, QImage] = {}
        self.items: dict[str, QListWidgetItem] = {}
        self.reasons: dict[str, list[tuple[str, str]]] = {}
        self.overrides: dict[str, bool] = {}    # True=失敗扱い False=OK扱い
        self.last_moves: list[tuple[str, str]] = []
        self.last_created_dirs: set[str] = set()
        self.worker: AnalyzeWorker | None = None
        self.pending: list[str] = []

        self._reclass_timer = QTimer(self, singleShot=True, interval=150)
        self._reclass_timer.timeout.connect(self.reclassify)
        # 解析結果は届くたびに描くと重いので、0.3 秒ごとにまとめて一覧へ追加する
        self._incoming: list[tuple[az.Metrics, QImage | None]] = []
        self._flush_timer = QTimer(self, interval=300)
        self._flush_timer.timeout.connect(self._flush_results)
        self._deco_key: dict[str, tuple] = {}   # サムネイルの描き直しが必要かどうかの判定用

        self._build_toolbar()
        self._build_ui()
        self._update_actions()
        people.configure(self.cb_gpu.isChecked())
        QTimer.singleShot(0, self._show_backend)   # 起動後に GPU を確認して表示

    # ------------------------------------------------ UI 構築
    def _build_toolbar(self):
        tb = QToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        tb.setStyleSheet("QToolButton{padding:6px 10px;font-size:13px;}")
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            return a

        self.a_folder = act("📁 フォルダを開く", self.open_folder, "Ctrl+O", "フォルダ内の写真をまとめて読み込みます")
        self.a_files = act("🖼 写真を追加", self.open_files, "Ctrl+Shift+O")
        self.a_clear = act("リストを空にする", self.clear_all)
        tb.addSeparator()
        self.a_stop = act("■ 解析を中止", self.stop_analysis)
        tb.addSeparator()
        def menu_button(text, kind, tip):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            m = QMenu(b)
            m.addAction("移動する…", lambda: self.export_photos(kind, move=True))
            m.addAction("コピーする…", lambda: self.export_photos(kind, move=False))
            b.setMenu(m)
            tb.addWidget(b)
            return b

        self.b_fail = menu_button("✕ 失敗写真を移動・コピー ▾", "fail", "失敗写真を別のフォルダへ移動またはコピーします")
        self.b_ok = menu_button("✓ OK の写真を移動・コピー ▾", "ok", "OK の写真を別のフォルダへ移動またはコピーします")
        self.a_undo = act("↶ 移動を元に戻す", self.undo_move, "Ctrl+Z")
        self.a_csv = act("CSV に書き出し", self.export_csv)

    def _build_ui(self):
        split = QSplitter()
        split.addWidget(self._build_settings_panel())

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(230)
        self.filter_combo.addItem("失敗写真のみ", ("fail", None))
        self.filter_combo.addItem("すべて", ("all", None))
        self.filter_combo.addItem("OK の写真のみ", ("ok", None))
        self.filter_combo.insertSeparator(3)
        for code, label in az.REASONS.items():
            self.filter_combo.addItem(f"理由: {label}", ("reason", code))
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["ファイル名順", "撮影(更新)日時順", "ボケている順", "手ブレが強い順"])
        self.sort_combo.currentIndexChanged.connect(self.apply_sort)
        self.summary = QLabel("写真をドラッグ＆ドロップするか「フォルダを開く」で読み込んでください")
        self.summary.setStyleSheet("font-weight:bold;")
        bar.addWidget(QLabel("表示:"))
        bar.addWidget(self.filter_combo)
        bar.addWidget(QLabel("並び順:"))
        bar.addWidget(self.sort_combo)
        bar.addSpacing(12)
        bar.addWidget(self.summary, 1)
        cv.addLayout(bar)

        # タグ（写っている物）で絞り込み: 検索欄 + タグの一覧（クリックで選ぶ）
        tagrow = QHBoxLayout()
        tagrow.addWidget(QLabel("タグ:"))
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("タグを検索（例: サーフ、犬、car）")
        self.tag_search.setClearButtonEnabled(True)
        self.tag_search.textChanged.connect(self._refresh_tag_list)
        tagrow.addWidget(self.tag_search, 1)
        self.tag_mode = QComboBox()
        self.tag_mode.addItem("選んだタグのどれかが写っている", "any")
        self.tag_mode.addItem("選んだタグがすべて写っている", "all")
        self.tag_mode.addItem("選んだタグが写っていない", "none")
        self.tag_mode.currentIndexChanged.connect(self.apply_filter)
        tagrow.addWidget(self.tag_mode)
        clear = QPushButton("選択を解除")
        clear.clicked.connect(self._clear_tags)
        tagrow.addWidget(clear)
        cv.addLayout(tagrow)
        self.tag_list = QListWidget()
        self.tag_list.setFlow(QListWidget.Flow.LeftToRight)
        self.tag_list.setWrapping(True)
        self.tag_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.tag_list.setSpacing(2)
        self.tag_list.setMaximumHeight(64)
        self.tag_list.setToolTip("チェックしたタグで写真を絞り込みます。かっこ内は写っている写真の枚数です。")
        self.tag_list.itemChanged.connect(lambda _it: self.apply_filter())
        self.tag_items: dict[int, QListWidgetItem] = {}
        cv.addWidget(self.tag_list)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(ICON_W, ICON_H))
        self.list.setGridSize(QSize(ICON_W + 20, ICON_H + 44))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self.show_detail)
        self.list.itemDoubleClicked.connect(self.open_external)
        cv.addWidget(self.list, 1)

        hint = QLabel("ヒント: 選択して <b>F</b> = 失敗にする / <b>G</b> = OK にする / <b>R</b> = 自動判定に戻す　ダブルクリックで写真を開く")
        hint.setStyleSheet("color:#666;")
        cv.addWidget(hint)
        split.addWidget(center)

        split.addWidget(self._build_detail_panel())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([290, 760, 380])
        self.setCentralWidget(split)

        for key, val in (("F", True), ("G", False), ("R", None)):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(lambda _=False, v=val: self.set_override(v))
            self.addAction(a)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.backend_label = QLabel()
        self.statusBar().addPermanentWidget(self.backend_label)

    def _build_settings_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        t = self.th
        # (ウィジェット, Thresholds の項目名, 表示倍率) — 設定の保存・初期化はこの表で行う
        self._bindings: list[tuple[QWidget, str, float]] = []

        def spin(form, label, name, lo, hi, step, tip, scale=1.0, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(0)
            s.setValue(getattr(t, name) * scale)
            s.setToolTip(tip)
            if suffix:
                s.setSuffix(suffix)
            s.valueChanged.connect(self._settings_changed)
            form.addRow(label, s)
            self._bindings.append((s, name, scale))
            return s

        def check(form, label, name, tip=""):
            c = QCheckBox(label)
            c.setChecked(getattr(t, name))
            c.setToolTip(tip)
            c.toggled.connect(self._settings_changed)
            form.addRow(c)
            self._bindings.append((c, name, 1.0))
            return c

        g = QGroupBox("処理速度")
        f = QFormLayout(g)
        self.cb_gpu = QCheckBox("グラフィックボード（GPU）を使う")
        self.cb_gpu.setToolTip("物体（タグ）検出の AI を GPU で動かして速くします（DirectML。NVIDIA / AMD / Intel 対応）。\n"
                               "うまく動かないときはチェックを外すと CPU で処理します。")
        self.cb_gpu.setChecked(self.settings.value("gpu", True) in (True, "true", "1", 1))
        self.cb_gpu.toggled.connect(self._gpu_changed)
        f.addRow(self.cb_gpu)
        v.addWidget(g)

        g = QGroupBox("ピンボケ")
        f = QFormLayout(g)
        spin(f, "鮮鋭度のしきい値", "blur", 1, 2000, 5,
             "鮮鋭度がこの値より低い写真をボケと判定します。大きくするほど厳しくなります。")
        self.cb_rel = check(f, "読み込んだ写真の中で\n相対的にボケている写真も検出", "relative_blur")
        self.sp_rel = spin(f, "中央値に対する割合", "relative_ratio", 5, 90, 5,
                           "全体の中央値の何 % 未満をボケとするか", 100, " %")
        v.addWidget(g)

        g = QGroupBox("手ブレ")
        f = QFormLayout(g)
        self.cb_motion = check(f, "手ブレを検出", "check_motion",
                               "画面全体が同じ方向に流れている写真を手ブレと判定します")
        self.sp_motion = spin(f, "手ブレらしさのしきい値", "motion", 10, 95, 1,
                              "手ブレらしさ（0〜100）がこの値を超えたら手ブレと判定します。\n"
                              "小さくするほど軽いブレも拾いますが、誤判定が増えます。", 100)
        v.addWidget(g)

        g = QGroupBox("被写体")
        f = QFormLayout(g)
        self.cb_subject = check(f, "被写体が写っている写真は\n被写体のピントで判定する", "check_subject",
                                "背景がボケていても被写体にピントが合っていれば OK にします。\n"
                                "被写体がボケていて背景にピントが合っている写真は失敗にします。")
        self.subject_combo = QComboBox()
        self.subject_combo.setEditable(True)            # 文字を入力して探せる
        self.subject_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for key, ja in az.subject_choices():
            en = "" if key == az.SUBJECT_AUTO else f"  {key}"
            self.subject_combo.addItem(ja + en, key)
        comp = self.subject_combo.completer()
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setCompletionMode(comp.CompletionMode.PopupCompletion)
        self.subject_combo.setCurrentIndex(max(0, self.subject_combo.findData(t.subject)))
        self.subject_combo.setToolTip("ピントを判定する被写体の種類。「自動」は写真ごとに一番大きく写っているものを被写体にします。\n"
                                      "文字を入力すると候補を探せます（例: サーフ、犬、car）。")
        self.subject_combo.currentIndexChanged.connect(self._settings_changed)
        f.addRow("被写体の種類", self.subject_combo)
        self.sp_face = spin(f, "顔の鮮鋭度のしきい値", "face_blur", 5, 2000, 5,
                            "被写体が人物のとき、顔の鮮鋭度がこれ未満なら「ピントが合っていない」と判定します")
        self.sp_bg = spin(f, "背景に対する被写体の割合", "person_bg_ratio", 5, 95, 5,
                          "被写体のくっきり度が背景のこの割合未満なら「背景にピントが合っている」と判定します", 100, " %")
        self.cb_eyes = check(f, "目つぶりの可能性を検出\n（正面を向いた大きめの顔のみ）", "check_eyes")
        v.addWidget(g)

        g = QGroupBox("明るさ")
        f = QFormLayout(g)
        spin(f, "暗すぎ（平均輝度 <）", "dark", 0, 128, 1, "平均輝度(0〜255)がこれ未満なら暗すぎ")
        spin(f, "明るすぎ（平均輝度 >）", "bright", 128, 255, 1, "平均輝度(0〜255)がこれを超えたら明るすぎ")
        spin(f, "白飛び（面積 >）", "highlight_clip", 1, 100, 1, "真っ白な画素の割合がこれを超えたら白飛び", 100, " %")
        spin(f, "黒つぶれ（面積 >）", "shadow_clip", 1, 100, 1, "真っ黒な画素の割合がこれを超えたら黒つぶれ", 100, " %")
        spin(f, "コントラスト不足（<）", "low_contrast", 0, 80, 1, "明暗差(標準偏差)がこれ未満ならコントラスト不足")
        v.addWidget(g)

        g = QGroupBox("その他")
        f = QFormLayout(g)
        self.cb_sim = check(f, "連写・似た写真のうち\nベスト以外を失敗にする", "check_similar")
        self.sp_simd = spin(f, "類似とみなす差（0〜30）", "similar_distance", 0, 30, 1,
                            "大きいほど「似ている」とみなす範囲が広がります")
        v.addWidget(g)

        reset = QPushButton("判定基準を初期値に戻す")
        reset.clicked.connect(self.reset_thresholds)
        v.addWidget(reset)
        v.addStretch(1)
        self._update_enabled()

        sc = QScrollArea()
        sc.setWidget(w)
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setMinimumWidth(w.sizeHint().width() + 24)
        return sc

    def _build_detail_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.preview = QLabel("写真を選択すると\nここに表示されます")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 260)
        self.preview.setStyleSheet("background:#222;color:#aaa;")
        v.addWidget(self.preview, 3)
        self.info = QTextBrowser()
        v.addWidget(self.info, 2)
        row = QHBoxLayout()
        for text, val in (("失敗にする (F)", True), ("OK にする (G)", False), ("自動判定 (R)", None)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, x=val: self.set_override(x))
            row.addWidget(b)
        v.addLayout(row)
        return w

    # ------------------------------------------------ しきい値
    def _load_thresholds(self) -> az.Thresholds:
        t = az.Thresholds()
        if self.settings.value("th_version") != az.THRESHOLDS_VERSION:
            self.settings.remove("th")
            self.settings.setValue("th_version", az.THRESHOLDS_VERSION)
            return t
        for f in fields(t):
            val = self.settings.value(f"th/{f.name}")
            if val is None:
                continue
            try:
                if f.type in ("bool", bool):
                    val = val in (True, "true", "1", 1)
                elif f.type in ("int", int):
                    val = int(val)
                else:
                    val = float(val)
                setattr(t, f.name, val)
            except (TypeError, ValueError):
                pass
        return t

    def _settings_changed(self, *_):
        t = self.th
        for wdg, name, scale in self._bindings:
            if isinstance(wdg, QCheckBox):
                setattr(t, name, wdg.isChecked())
            elif isinstance(getattr(az.Thresholds(), name), int):
                setattr(t, name, int(round(wdg.value() / scale)))
            else:
                setattr(t, name, wdg.value() / scale)
        key = self.subject_combo.currentData()
        if key:
            t.subject = key
        self._update_enabled()
        for k, val in asdict(t).items():
            self.settings.setValue(f"th/{k}", val)
        self._reclass_timer.start()

    def _gpu_changed(self, on: bool):
        self.settings.setValue("gpu", on)
        people.configure(on)
        self._show_backend()

    def _show_backend(self):
        try:
            kind, note = people.backend()
        except Exception as e:
            kind, note = "opencv", str(e)
        text = {"gpu": "AI処理: GPU", "cpu": "AI処理: CPU", "opencv": "AI処理: CPU（簡易）"}.get(kind, "")
        self.backend_label.setText(text + ("　⚠" if note else ""))
        self.backend_label.setToolTip(note or ("グラフィックボードで物体（タグ）検出をしています（DirectML）"
                                               if kind == "gpu" else "CPU で物体（タグ）検出をしています"))

    def _update_enabled(self):
        self.sp_rel.setEnabled(self.cb_rel.isChecked())
        self.sp_motion.setEnabled(self.cb_motion.isChecked())
        self.sp_face.setEnabled(self.cb_subject.isChecked())
        self.sp_bg.setEnabled(self.cb_subject.isChecked())
        self.subject_combo.setEnabled(self.cb_subject.isChecked())
        self.sp_simd.setEnabled(self.cb_sim.isChecked())

    def reset_thresholds(self):
        d = az.Thresholds()
        for wdg, name, scale in self._bindings:
            wdg.blockSignals(True)
            if isinstance(wdg, QCheckBox):
                wdg.setChecked(getattr(d, name))
            else:
                wdg.setValue(getattr(d, name) * scale)
            wdg.blockSignals(False)
        self.subject_combo.blockSignals(True)
        self.subject_combo.setCurrentIndex(max(0, self.subject_combo.findData(d.subject)))
        self.subject_combo.blockSignals(False)
        self._settings_changed()

    # ------------------------------------------------ 読み込み
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "写真のフォルダを選択", self.settings.value("lastdir", ""))
        if d:
            self.settings.setValue("lastdir", d)
            self.add_paths([d])

    def open_files(self):
        exts = " ".join(f"*{e}" for e in sorted(az.SUPPORTED_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, "写真を選択", self.settings.value("lastdir", ""),
                                                f"画像 ({exts})")
        if files:
            self.settings.setValue("lastdir", str(Path(files[0]).parent))
            self.add_paths(files)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths: list[str]):
        imgs = [p for p in az.list_images(paths) if p not in self.metrics and p not in self.pending]
        if not imgs:
            self.statusBar().showMessage("新しく読み込む写真がありませんでした", 5000)
            return
        if self.worker and self.worker.isRunning():
            self.pending.extend(imgs)
            return
        self._start(imgs)

    def _start(self, imgs: list[str]):
        self.progress.setRange(0, len(imgs))
        self.progress.setValue(0)
        self.progress.show()
        self.statusBar().showMessage(f"{len(imgs)} 枚を解析中…")
        self.worker = AnalyzeWorker(imgs)
        self.worker.result.connect(self._on_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._flush_timer.start()
        self._update_actions()

    def _on_result(self, m: az.Metrics, qimg: QImage | None):
        if m.error != "中止":
            self._incoming.append((m, qimg))   # 実際の追加は _flush_results でまとめて行う

    def _flush_results(self):
        """届いた解析結果をまとめて一覧に追加する。新しい写真だけを判定・描画するので、
        枚数が増えても 1 回あたりの処理は軽いまま。全体の判定し直しは解析の最後に 1 回だけ。"""
        if not self._incoming:
            return
        batch, self._incoming = self._incoming, []
        median = self._median_sharpness()
        self.list.setUpdatesEnabled(False)
        try:
            for m, qimg in batch:
                p = m.path
                self.metrics[p] = m
                if qimg is None:
                    qimg = QImage(ICON_W, ICON_H * 3 // 4, QImage.Format.Format_RGB888)
                    qimg.fill(QColor("#555"))
                self.thumbs[p] = qimg
                self.reasons[p] = az.judge(m, self.th, *median).reasons
                it = QListWidgetItem(Path(p).name)
                it.setData(ROLE_PATH, p)
                self.list.addItem(it)
                self.items[p] = it
                self._decorate(p)
                it.setHidden(not self._visible(p))
        finally:
            self.list.setUpdatesEnabled(True)
        self._update_counts()

    def _on_progress(self, done, total):
        self.progress.setValue(done)
        self.statusBar().showMessage(f"解析中… {done} / {total}")

    def _on_finished(self):
        self._flush_results()
        self._flush_timer.stop()
        self.progress.hide()
        self.worker = None
        if self.pending:
            imgs, self.pending = self.pending, []
            self._start(imgs)
            return
        self.reclassify()
        self.apply_sort()
        self.statusBar().showMessage("解析が完了しました", 5000)
        self._update_actions()

    def stop_analysis(self):
        self.pending.clear()
        if self.worker:
            self.worker.stop()
            self.statusBar().showMessage("中止しています…")

    def clear_all(self):
        if self.worker and self.worker.isRunning():
            return
        self.list.clear()
        for d in (self.metrics, self.thumbs, self.items, self.reasons, self.overrides, self._deco_key):
            d.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("写真を選択すると\nここに表示されます")
        self.info.clear()
        self.reclassify()

    # ------------------------------------------------ 判定
    def is_failed(self, p: str) -> bool:
        if p in self.overrides:
            return self.overrides[p]
        return bool(self.reasons.get(p))

    def _median_sharpness(self) -> tuple[float | None, dict[int, float]]:
        """(写真全体の鮮鋭度の中央値, 被写体の種類ごとの鮮鋭度の中央値)。5 枚未満なら使わない。"""
        ok_sharp = [m.sharpness for m in self.metrics.values() if m.ok]
        return (statistics.median(ok_sharp) if len(ok_sharp) >= 5 else None,
                az.subject_medians(self.metrics.values(), self.th.subject))

    def reclassify(self):
        median = self._median_sharpness()
        losers = az.similar_losers(list(self.metrics.values()), self.th.similar_distance) \
            if self.th.check_similar else {}
        for p, m in self.metrics.items():
            v = az.judge(m, self.th, *median)
            if p in losers:
                v.add("similar", f"ベスト: {losers[p]}")
            self.reasons[p] = v.reasons
            self._decorate(p)
        self.apply_filter()
        self.show_detail()

    def reason_texts(self, p: str) -> list[str]:
        return [text for _, text in self.reasons.get(p) or []]

    def _decorate(self, p: str):
        it = self.items.get(p)
        if it is None:
            return
        failed = self.is_failed(p)
        base = self.thumbs[p]
        m = self.metrics[p]
        sub = az.pick_subject(m, self.th.subject) if (m.ok and self.th.check_subject) else None
        badge = sub.label if sub else ""
        key = (failed, self.overrides.get(p), badge, tuple(self.reasons.get(p) or ()))
        if self._deco_key.get(p) == key:
            return
        self._deco_key[p] = key
        pm = QPixmap(ICON_W, ICON_H)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        x = (ICON_W - base.width()) // 2
        y = (ICON_H - base.height()) // 2
        painter.drawImage(x, y, base)
        if failed:
            painter.setPen(QPen(QColor("#e53935"), 6))
            painter.drawRect(x + 3, y + 3, base.width() - 6, base.height() - 6)
            label = "✕ 失敗"
            if p in self.overrides:
                label += "（手動）"
            color = QColor("#e53935")
        else:
            label = "✓ OK" + ("（手動）" if p in self.overrides else "")
            color = QColor("#2e7d32")
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        painter.setFont(f)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        painter.fillRect(x + 4, y + 4, tw, fm.height() + 4, color)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(x + 9, y + 6 + fm.ascent(), label)
        if badge:   # 被写体の種類
            bw = fm.horizontalAdvance(badge) + 10
            painter.fillRect(x + base.width() - bw - 4, y + 4, bw, fm.height() + 4, QColor("#1565c0"))
            painter.drawText(x + base.width() - bw + 1, y + 6 + fm.ascent(), badge)
        painter.end()
        it.setIcon(QIcon(pm))
        reasons = self.reason_texts(p)
        sub = " / ".join(az.REASONS[c] for c, _ in self.reasons.get(p, [])) if failed else ""
        it.setText(Path(p).name + (f"\n{sub}" if sub else ""))
        it.setForeground(QColor("#c62828") if failed else QColor("#222"))
        tags = "、".join(people.LABELS_JA[c] for c in sorted(m.tags)) if m.ok else ""
        it.setToolTip(p + ("\n" + "\n".join(reasons) if reasons else "") + (f"\nタグ: {tags}" if tags else ""))

    def _visible(self, p: str) -> bool:
        kind, code = self.filter_combo.currentData() or ("fail", None)
        failed = self.is_failed(p)
        if kind == "fail":
            show = failed
        elif kind == "ok":
            show = not failed
        elif kind == "reason":
            show = any(c == code for c, _ in self.reasons.get(p, []))
        else:
            show = True
        sel = self._selected_tags()
        if show and sel:
            tags = self.metrics[p].tags if self.metrics[p].ok else set()
            mode = self.tag_mode.currentData()
            if mode == "all":
                show = sel <= tags
            elif mode == "none":
                show = not (sel & tags)
            else:
                show = bool(sel & tags)
        return show

    # ------------------------------------------------ タグ
    def _selected_tags(self) -> set[int]:
        return {c for c, it in self.tag_items.items() if it.checkState() == Qt.CheckState.Checked}

    def _clear_tags(self):
        self.tag_list.blockSignals(True)
        for it in self.tag_items.values():
            it.setCheckState(Qt.CheckState.Unchecked)
        self.tag_list.blockSignals(False)
        self.apply_filter()

    def _refresh_tag_list(self, *_):
        """読み込んだ写真に写っているタグを、枚数の多い順に並べる。検索欄の文字で絞り込む。"""
        counts: dict[int, int] = {}
        for m in self.metrics.values():
            if m.ok:
                for c in m.tags:
                    counts[c] = counts.get(c, 0) + 1
        q = self.tag_search.text()
        self.tag_list.blockSignals(True)
        try:
            for c, n in counts.items():
                it = self.tag_items.get(c)
                if it is None:
                    it = QListWidgetItem()
                    it.setData(ROLE_PATH, c)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Unchecked)
                    self.tag_list.addItem(it)
                    self.tag_items[c] = it
                it.setText(f"{people.LABELS_JA[c]} ({n})")
                it.setToolTip(people.LABELS_EN[c])
            for c, it in self.tag_items.items():
                n = counts.get(c, 0)
                # 選んだタグは、検索で外れても見えるように残す
                checked = it.checkState() == Qt.CheckState.Checked
                it.setHidden(not checked and (n == 0 or not tag_matches(c, q)))
            order = sorted(self.tag_items, key=lambda c: (-counts.get(c, 0), c))
            if [self.tag_list.item(i).data(ROLE_PATH) for i in range(self.tag_list.count())] != order:
                for row in range(self.tag_list.count() - 1, -1, -1):
                    self.tag_list.takeItem(row)
                for c in order:
                    self.tag_list.addItem(self.tag_items[c])
        finally:
            self.tag_list.blockSignals(False)

    def apply_filter(self):
        self.list.setUpdatesEnabled(False)
        try:
            for p, it in self.items.items():
                it.setHidden(not self._visible(p))
        finally:
            self.list.setUpdatesEnabled(True)
        self._update_counts()

    def _update_counts(self):
        n_fail = n_shown = 0
        counts = {c: 0 for c in az.REASONS}
        for p, it in self.items.items():
            n_fail += self.is_failed(p)
            n_shown += not it.isHidden()
            for c, _ in self.reasons.get(p, []):
                counts[c] += 1
        # 選択肢に件数を表示する
        for i in range(self.filter_combo.count()):
            d = self.filter_combo.itemData(i)
            if d and d[0] == "reason":
                self.filter_combo.setItemText(i, f"理由: {az.REASONS[d[1]]}（{counts[d[1]]}）")
        n = len(self.items)
        self._refresh_tag_list()
        if n:
            self.summary.setText(f"全 {n} 枚中　失敗 {n_fail} 枚　/　OK {n - n_fail} 枚　（表示中 {n_shown} 枚）")
        else:
            self.summary.setText("写真をドラッグ＆ドロップするか「フォルダを開く」で読み込んでください")
        self._update_actions()

    def apply_sort(self):
        mode = self.sort_combo.currentIndex()
        keys = {
            0: lambda p: Path(p).name.lower(),
            1: lambda p: self.metrics[p].mtime,
            2: lambda p: self.metrics[p].sharpness,
            3: lambda p: -self.metrics[p].motion_score,
        }
        order = sorted(self.items, key=keys[mode])
        selected = {it.data(ROLE_PATH) for it in self.list.selectedItems()}
        self.list.setUpdatesEnabled(False)
        try:
            for row in range(self.list.count() - 1, -1, -1):   # 後ろから外す（前から外すと遅い）
                self.list.takeItem(row)
            for p in order:
                self.list.addItem(self.items[p])
                if p in selected:
                    self.items[p].setSelected(True)
        finally:
            self.list.setUpdatesEnabled(True)
        self.apply_filter()

    def set_override(self, value: bool | None):
        for it in self.list.selectedItems():
            p = it.data(ROLE_PATH)
            if value is None:
                self.overrides.pop(p, None)
            else:
                self.overrides[p] = value
            self._decorate(p)
        self.apply_filter()
        self.show_detail()

    def open_external(self, it):
        try:
            os.startfile(it.data(ROLE_PATH))
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"写真を開けませんでした。\n{e}")

    # ------------------------------------------------ 詳細表示
    def show_detail(self):
        sel = self.list.selectedItems()
        if not sel:
            return
        p = sel[0].data(ROLE_PATH)
        m = self.metrics.get(p)
        if m is None:
            return
        if getattr(self, "_preview_path", None) != p:
            self._preview_path = p
            try:
                img = az.load_image(p, 1000)
                self._preview_pm = QPixmap.fromImage(pil_to_qimage(img))
            except Exception:
                self._preview_pm = QPixmap()
            self._preview_base = self._preview_pm
        if not self._preview_pm.isNull():
            try:
                self._preview_pm = self._draw_objects(self._preview_base.copy(), m)
            except Exception:
                self._preview_pm = self._preview_base   # 重ね描きに失敗しても写真は表示する
        self._fit_preview()

        failed = self.is_failed(p)
        reasons = self.reason_texts(p)
        state = "<span style='color:#c62828'>✕ 失敗</span>" if failed else "<span style='color:#2e7d32'>✓ OK</span>"
        if p in self.overrides:
            state += "（手動で設定）"
        html = [f"<h3>{state}</h3>", f"<b>{Path(p).name}</b><br><small>{p}</small><br>"]
        if reasons:
            html.append("<b>自動判定の理由</b><ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>")
        if m.ok:
            t = self.th
            sub = az.pick_subject(m, t.subject) if t.check_subject else None
            html.append("<b>測定値</b><table cellspacing=3>")
            rows = [
                ("サイズ", f"{m.width} × {m.height}", ""),
                ("鮮鋭度（全体）", f"{m.sharpness:.0f}", f"しきい値 {t.blur:.0f}"),
                ("手ブレらしさ", f"{m.motion_score * 100:.0f}",
                 f"しきい値 {t.motion * 100:.0f}　方向: {az._angle_name(m.motion_angle)}"
                 + (f"　被写体部分 {sub.motion * 100:.0f}" if sub and sub.motion >= 0 else "")),
            ]
            tags = "、".join(people.LABELS_JA[c] for c in sorted(m.tags)) or "なし"
            rows.append(("タグ", tags, ""))
            if sub:
                how = "自動（一番大きいもの）" if t.subject == az.SUBJECT_AUTO else "設定で指定"
                rows.append(("被写体", f"{sub.label}（{len(sub.objs)}）", how))
                bg = f"背景 {sub.background:.0f}" if sub.background > 0 else "背景なし"
                rows.append(("被写体の鮮鋭度", f"{sub.sharpness:.0f}", bg))
                if sub.face_sharpness >= 0:
                    rows.append(("顔の鮮鋭度", f"{sub.face_sharpness:.0f}",
                                 f"しきい値 {t.face_blur:.0f}　明暗差比 {sub.face_rel:.0f}"))
                elif sub.cls == people.PERSON and m.faces:
                    rows.append(("顔の鮮鋭度", "—", "顔が小さいため人物全体で判定"))
            else:
                rows.append(("被写体", "なし", "写真全体で判定"))
            if m.eyes_checked:
                rows.append(("目の判定", f"{m.eyes_checked} 人", f"目が未検出 {m.eyes_missing}"))
            rows += [
                ("平均輝度", f"{m.brightness:.0f}", f"{t.dark:.0f}〜{t.bright:.0f}"),
                ("コントラスト", f"{m.contrast:.1f}", f"{t.low_contrast:.0f} 以上"),
                ("白飛び面積", f"{m.highlight_clip * 100:.1f} %",
                 (f"被写体部分 {sub.highlight * 100:.0f} %" if sub and sub.highlight >= 0
                  else f"下 6 割 {m.highlight_clip_lower * 100:.0f} %") + f"　しきい値 {t.highlight_clip * 100:.0f} %"),
                ("黒つぶれ面積", f"{m.shadow_clip * 100:.1f} %", f"{t.shadow_clip * 100:.0f} % 以下"),
            ]
            for a, b, c in rows:
                html.append(f"<tr><td>{a}</td><td align=right><b>{b}</b></td><td style='color:#777'>{c}</td></tr>")
            html.append("</table>")
            if m.objects or m.faces:
                html.append("<p style='color:#777'><small>プレビュー: 緑 = 被写体、灰 = ほかに見つかった物、黄 = 顔</small></p>")
        else:
            html.append(f"<p>{m.error}</p>")
        if len(sel) > 1:
            html.append(f"<p>（{len(sel)} 枚選択中）</p>")
        self.info.setHtml("".join(html))

    def _draw_objects(self, pm: QPixmap, m: az.Metrics) -> QPixmap:
        """プレビューに、被写体の形（緑）・ほかの物の枠と名前（灰）・顔（黄）を重ねる"""
        if pm.isNull() or not m.ok or not (m.objects or m.faces):
            return pm
        sub = az.pick_subject(m, self.th.subject) if self.th.check_subject else None
        subj_ids = {id(o) for o in sub.objs} if sub else set()
        W, H = pm.width(), pm.height()
        painter = QPainter(pm)
        lw = max(2, W // 300)
        f = QFont()
        f.setPointSize(max(8, W // 90))
        painter.setFont(f)
        for o in m.objects:
            if o.score < az.TAG_MIN_SCORE or id(o) in subj_ids:
                continue
            x, y, w, h = o.px(W, H)
            painter.setPen(QPen(QColor(200, 200, 200, 200), max(1, lw // 2)))
            painter.drawRect(x, y, w, h)
            painter.drawText(x + 3, y + painter.fontMetrics().ascent() + 2, o.label)
        for o in (sub.objs if sub else []):
            mk = o.small_mask()
            if mk is not None:
                rgba = np.zeros(mk.shape + (4,), np.uint8)
                rgba[mk] = (76, 175, 80, 110)
                img = QImage(rgba.data, mk.shape[1], mk.shape[0], mk.shape[1] * 4, QImage.Format.Format_RGBA8888)
                painter.drawImage(pm.rect(), img.copy())
            x, y, w, h = o.px(W, H)
            painter.setPen(QPen(QColor("#4caf50"), lw))
            painter.drawRect(x, y, w, h)
        painter.setPen(QPen(QColor("#ffd600"), lw))
        for b in m.faces:
            x, y, w, h = b.px(W, H)
            painter.drawRect(x, y, w, h)
        painter.end()
        return pm

    def _fit_preview(self):
        pm = getattr(self, "_preview_pm", None)
        if pm is None or pm.isNull():
            return
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_preview()

    # ------------------------------------------------ 書き出し
    def failed_paths(self) -> list[str]:
        return [p for p in self.items if self.is_failed(p)]

    def ok_paths(self) -> list[str]:
        return [p for p in self.items if not self.is_failed(p)]

    def export_photos(self, kind: str, move: bool):
        """kind="fail" なら失敗写真、"ok" なら OK の写真を移動/コピーする。"""
        paths = self.failed_paths() if kind == "fail" else self.ok_paths()
        label = "失敗写真" if kind == "fail" else "OK の写真"
        if not paths:
            QMessageBox.information(self, APP_NAME, f"{label}はありません。")
            return
        dlg = ExportDialog(self, kind, move, len(paths), self.settings)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dest = dlg.destination()          # None = 元フォルダの中に作る
        sub = az.OUTPUT_DIR_NAME if kind == "fail" else az.OK_DIR_NAME
        verb = "移動" if move else "コピー"

        with_raw = dlg.with_raw()
        moves, errors, skipped, created = [], [], [], set()
        moved_images, n_raw = [], 0
        raw_index: dict = {}
        for p in paths:
            src = Path(p)
            dst_dir = Path(dest) if dest else src.parent / sub
            # 一緒に動かすファイル: 写真本体 + 同じ名前の RAW / 編集情報(.xmp)
            group = [src] + (az.companion_files(src, raw_index) if with_raw else [])
            try:
                if dst_dir.resolve() == src.parent.resolve():
                    skipped.append(src.name)      # 移動先が元と同じフォルダ
                    continue
                if not dst_dir.exists():
                    # 新しく作るフォルダ（途中の親フォルダも含む）を覚えておき、元に戻すときに消す
                    d = dst_dir
                    while not d.exists():
                        created.add(str(d))
                        d = d.parent
                    dst_dir.mkdir(parents=True)
                # 同じ名前があれば上書きせず番号を付ける（JPG と RAW は同じ番号にそろえる）
                stem, i = src.stem, 1
                while any((dst_dir / (stem + f.suffix)).exists() for f in group):
                    stem = f"{src.stem}_{i}"
                    i += 1
                for f in group:
                    dst = dst_dir / (stem + f.suffix)
                    if move:
                        shutil.move(str(f), str(dst))
                        moves.append((str(f), str(dst)))
                    else:
                        shutil.copy2(str(f), str(dst))
                    n_raw += f is not src
                    if f is src and move:
                        moved_images.append(str(src))
            except OSError as e:
                errors.append(f"{src.name}: {e}")
        if move:
            self.last_moves = moves
            self.last_created_dirs = created
            for src in moved_images:
                self._remove_item(src)
            self.apply_filter()
        done = len(paths) - len(errors) - len(skipped)
        where = f"「{dest}」" if dest else f"各フォルダ内の「{sub}」"
        msg = f"{label} {done} 枚を{where}へ{verb}しました。" if done else f"{verb}した写真はありません。"
        if n_raw:
            msg += f"\n（同じ名前の RAW などのファイル {n_raw} 個も一緒に{verb}しました）"
        if skipped:
            msg += f"\n\n{verb}先が元と同じフォルダなので{verb}しなかったもの: {len(skipped)} 枚"
        if errors:
            msg += "\n\nできなかったもの:\n" + "\n".join(errors[:20])
        QMessageBox.information(self, APP_NAME, msg)
        self._update_actions()

    def _remove_item(self, p: str):
        it = self.items.pop(p, None)
        if it is not None:
            self.list.takeItem(self.list.row(it))
        for d in (self.metrics, self.thumbs, self.reasons, self.overrides, self._deco_key):
            d.pop(p, None)

    def undo_move(self):
        if not self.last_moves:
            return
        restored, errors = [], []
        for src, dst in reversed(self.last_moves):
            try:
                if Path(src).exists():
                    raise OSError("元の場所に同名ファイルがあります")
                shutil.move(dst, src)
                restored.append(src)
            except OSError as e:
                errors.append(f"{Path(dst).name}: {e}")
        self.last_moves = []
        # このアプリが作ったフォルダで、空になったものだけ削除する（深い順に）
        for d in sorted(self.last_created_dirs, key=len, reverse=True):
            try:
                os.rmdir(d)
            except OSError:
                pass
        self.last_created_dirs = set()
        msg = f"{len(restored)} 個のファイルを元に戻しました。"
        if errors:
            msg += "\n\n戻せなかったもの:\n" + "\n".join(errors[:20])
        QMessageBox.information(self, APP_NAME, msg)
        self.add_paths(restored)
        self._update_actions()

    def export_csv(self):
        if not self.items:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "CSV に書き出し", "判定結果.csv", "CSV (*.csv)")
        if not fn:
            return
        # 拡張子を必ず .csv にして、写真など他のファイルを上書きしないようにする
        if Path(fn).suffix.lower() != ".csv":
            fn += ".csv"
            if Path(fn).exists() and QMessageBox.question(
                    self, APP_NAME, f"{Path(fn).name} は既にあります。上書きしますか？"
            ) != QMessageBox.StandardButton.Yes:
                return
        try:
            self._write_csv(fn)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"CSV を保存できませんでした。\n（Excel などで開いていると保存できません）\n\n{e}")
            return
        self.statusBar().showMessage(f"書き出しました: {fn}", 6000)

    def _write_csv(self, fn: str):
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["ファイル", "判定", "手動", "理由", "鮮鋭度", "手ブレらしさ", "ブレ方向",
                        "タグ", "被写体", "顔の数", "顔の鮮鋭度", "被写体の鮮鋭度", "背景の鮮鋭度",
                        "平均輝度", "コントラスト", "白飛び%", "黒つぶれ%", "目未検出"])
            for p in self.items:
                m = self.metrics[p]
                sub = az.pick_subject(m, self.th.subject) if (m.ok and self.th.check_subject) else None
                w.writerow([p, "失敗" if self.is_failed(p) else "OK", "○" if p in self.overrides else "",
                            " / ".join(self.reason_texts(p)), f"{m.sharpness:.1f}",
                            f"{m.motion_score * 100:.0f}", az._angle_name(m.motion_angle),
                            "、".join(people.LABELS_JA[c] for c in sorted(m.tags)) if m.ok else "",
                            sub.label if sub else "", len(m.faces),
                            f"{sub.face_sharpness:.1f}" if sub and sub.face_sharpness >= 0 else "",
                            f"{sub.sharpness:.1f}" if sub else "",
                            f"{sub.background:.1f}" if sub and sub.background > 0 else "",
                            f"{m.brightness:.1f}", f"{m.contrast:.1f}", f"{m.highlight_clip * 100:.2f}",
                            f"{m.shadow_clip * 100:.2f}", m.eyes_missing])

    def _update_actions(self):
        busy = self.worker is not None
        has = bool(self.items)
        self.a_stop.setEnabled(busy)
        self.a_clear.setEnabled(has and not busy)
        self.b_fail.setEnabled(has and not busy)
        self.b_ok.setEnabled(has and not busy)
        self.a_csv.setEnabled(has)
        self.a_undo.setEnabled(bool(self.last_moves) and not busy)

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        super().closeEvent(e)


def diagnose(out_path: str, photo: str | None):
    """PhotoCull.exe --diagnose 結果.txt [写真] : GPU が使えるかなどを調べてファイルに書く（不具合調査用）"""
    import time
    lines = [f"{APP_NAME} 診断  {datetime.now():%Y-%m-%d %H:%M:%S}"]
    try:
        import onnxruntime as ort
        lines.append(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}")
    except Exception as e:
        lines.append(f"onnxruntime なし: {e}")
    lines.append(f"models available={people.available()}")
    for gpu in (True, False):
        people.configure(gpu)
        kind, note = people.backend()
        lines.append(f"GPU設定={'オン' if gpu else 'オフ'} → 処理={kind} {note}")
        if photo:
            az.analyze(photo)
            t = time.perf_counter()
            m, _ = az.analyze(photo)
            lines.append(f"  1枚の解析 {time.perf_counter() - t:.3f}s  人物={len(m.persons)} 顔={len(m.faces)} "
                         f"判定={[t for _, t in az.judge(m, az.Thresholds()).reasons]}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--diagnose":
        diagnose(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    app = QApplication(sys.argv)
    install_crash_handlers()
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    f = app.font()
    f.setFamily("Yu Gothic UI")
    f.setPointSize(10)
    app.setFont(f)
    win = MainWindow()
    win.show()
    args = [a for a in sys.argv[1:] if os.path.exists(a)]
    if args:
        win.add_paths(args)
    sys.exit(app.exec())
