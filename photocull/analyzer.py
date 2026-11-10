"""画像解析エンジン: 失敗写真の判定に使う各種指標を計算する。

計算した指標 (Metrics) と判定しきい値 (Thresholds) を分けているので、
しきい値を変えても再解析せずに即座に判定し直せる。
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import people

# 1億画素を超える画像も読めるようにする（初期値の約1.8億だと警告で落ちることがある）
Image.MAX_IMAGE_PIXELS = 400_000_000

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ANALYSIS_SIZE = 1024   # 全体の判定に使う長辺ピクセル数
DETAIL_SIZE = 2048     # 顔のピント判定に使う長辺ピクセル数
THUMB_SIZE = 220       # サムネイルの長辺ピクセル数


def _cascade_dir() -> str:
    # PyInstaller でまとめた場合は同梱した cascades フォルダを使う
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "cascades")
    return cv2.data.haarcascades


_cascade_xml: dict[str, str] = {}
_cascade_lock = threading.Lock()
_tls = threading.local()


def _load_cascade(name: str) -> cv2.CascadeClassifier:
    # OpenCV はファイルパスに日本語が含まれると読めないため、
    # Python で読み込んだ XML をメモリから渡す
    with _cascade_lock:
        if name not in _cascade_xml:
            try:
                with open(os.path.join(_cascade_dir(), name), encoding="utf-8") as f:
                    _cascade_xml[name] = f.read()
            except OSError:
                _cascade_xml[name] = ""
        xml = _cascade_xml[name]
    cc = cv2.CascadeClassifier()
    if xml:
        fs = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        cc.read(fs.getFirstTopLevelNode())
        fs.release()
    return cc


def _eye_cascade() -> cv2.CascadeClassifier:
    # CascadeClassifier は複数スレッドで共有すると壊れるので、スレッドごとに持つ
    cc = getattr(_tls, "eye_cascade", None)
    if cc is None:
        cc = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
        _tls.eye_cascade = cc
    return cc


# ------------------------------------------------------------------ 失敗の理由
# コード: 表示名。絞り込みの選択肢もこの順に並ぶ。
REASONS: dict[str, str] = {
    "blur": "ピンボケ",
    "motion": "手ブレ",
    "subject_blur": "被写体にピントが合っていない",
    "eyes": "目つぶりの可能性",
    "dark": "暗すぎ",
    "shadow": "黒つぶれ",
    "bright": "明るすぎ",
    "highlight": "白飛び",
    "lowcon": "コントラスト不足",
    "similar": "似た写真（ベスト以外）",
    "error": "読み込みエラー",
}


@dataclass
class Metrics:
    path: str
    ok: bool = True               # 読み込めたか
    error: str = ""
    width: int = 0
    height: int = 0
    sharpness: float = 0.0        # 最もピントが合っている領域の鮮鋭度
    sharpness_global: float = 0.0
    motion_score: float = 0.0     # 手ブレらしさ 0〜1（全体が同じ方向に流れているほど大きい）
    motion_angle: float = 0.0     # ブレの方向（度, 0=横, 90=縦）
    motion_perp: float = 0.0      # ブレと直角の方向に残っている細かさ（参考値）
    brightness: float = 0.0       # 平均輝度 0-255
    contrast: float = 0.0         # 輝度の標準偏差
    highlight_clip: float = 0.0   # 白飛び画素の割合（写真全体）
    highlight_clip_lower: float = 0.0  # 写真の下 6 割での白飛びの割合（明るい空を除いて見るため）
    shadow_clip: float = 0.0      # 黒つぶれ画素の割合
    objects: list = field(default_factory=list)   # 見つかった物 people.Obj（タグ・被写体の候補）
    faces: list = field(default_factory=list)     # 顔の枠 people.Box
    eyes_checked: int = 0         # 目の判定をした顔の数
    eyes_missing: int = 0         # 目が検出できなかった顔の数
    dhash: int = 0                # 類似判定用ハッシュ
    mtime: float = 0.0

    @property
    def persons(self) -> list:
        return [o for o in self.objects if o.cls == people.PERSON]

    @property
    def has_person(self) -> bool:
        return bool(self.persons or self.faces)

    @property
    def tags(self) -> set[int]:
        """写っている物の種類（タグ）。people.LABELS_JA の番号"""
        t = {o.cls for o in self.objects if o.score >= TAG_MIN_SCORE}
        if self.faces:
            t.add(people.PERSON)
        return t


# 判定基準の意味や初期値を変えたら上げる（古い保存設定を捨てて新しい初期値を使うため）
THRESHOLDS_VERSION = "4"


@dataclass
class Thresholds:
    blur: float = 60.0                # 鮮鋭度がこれ未満ならピンボケ
    relative_blur: bool = True        # 同じ一括処理内の中央値と比べても判定する
    relative_ratio: float = 0.50      # 中央値のこの割合未満ならボケ（人物は人物どうしで比べる）
    check_motion: bool = True
    motion: float = 0.65              # 手ブレらしさがこれを超えたら手ブレ
    check_subject: bool = True        # 被写体が写っていれば被写体のピントで判定する
    subject: str = "person"           # 被写体の種類（"auto" = 一番大きく写っているもの）
    face_blur: float = 15.0           # 顔の鮮鋭度がこれ未満なら人物ピンボケ
    person_bg_ratio: float = 0.35     # 被写体の鮮鋭度が背景のこの割合未満なら「背景にピント」
    dark: float = 40.0                # 平均輝度がこれ未満なら暗すぎ
    bright: float = 220.0             # 平均輝度がこれ超なら明るすぎ
    highlight_clip: float = 0.30      # 白飛び割合がこれ超なら白飛び
    shadow_clip: float = 0.60         # 黒つぶれ割合がこれ超なら黒つぶれ
    low_contrast: float = 15.0        # 標準偏差がこれ未満ならコントラスト不足
    check_eyes: bool = True
    check_similar: bool = False
    similar_distance: int = 8         # dHash のハミング距離がこれ以下なら類似


@dataclass
class Verdict:
    reasons: list[tuple[str, str]] = field(default_factory=list)   # (コード, 説明)

    def add(self, code: str, detail: str = ""):
        self.reasons.append((code, REASONS[code] + (f"（{detail}）" if detail else "")))

    @property
    def failed(self) -> bool:
        return bool(self.reasons)

    @property
    def codes(self) -> set[str]:
        return {c for c, _ in self.reasons}


_ORIENT = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def load_image(path: str, max_side: int, fit: bool = True) -> Image.Image:
    """fit=False のときは JPEG の縮小デコード（max_side 以上の大きさ）だけ行い、細かい縮小はしない。"""
    with Image.open(path) as src:
        try:
            orientation = src.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
        if src.format in ("JPEG", "MPO"):   # MPO = 一部のカメラが使う、プレビュー画像入りの JPEG
            # JPEG は縮小デコードで大幅に高速化・省メモリ化できる
            # 縮小デコードは「縦横とも指定以上」になる倍率を選ぶので、写真の縦横比に合わせて指定する
            # （正方形で指定すると 3:2 の写真では短辺が足りず、毎回フルサイズで読んでしまう）
            w0, h0 = src.size
            k = max_side / max(w0, h0)
            src.draft("RGB", (max(1, int(w0 * k)), max(1, int(h0 * k))))
        img = src
        if img.mode in ("I", "F") or img.mode.startswith("I;16"):
            # 16bit / 浮動小数の画像は 8bit に正規化する
            a = np.asarray(img, dtype=np.float32)
            lo, hi = np.percentile(a, (0.5, 99.5))
            a = np.clip((a - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
            img = Image.fromarray(a, "L")
        elif img.mode == "P":
            img = img.convert("RGBA")
        # 先に縮小してからメモリを使う処理をする（巨大な PNG/TIFF 対策）
        if fit:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
        else:
            # 縮小デコードできない形式（巨大な PNG など）でも、メモリを使いすぎない大きさにはしておく
            img.thumbnail((max_side * 2, max_side * 2), Image.Resampling.LANCZOS, reducing_gap=2.0)
        if img.mode in ("RGBA", "LA", "PA") or (img.mode != "RGB" and "A" in img.getbands()):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")  # L, CMYK, YCbCr など
        else:
            img = img.copy()   # ファイルを閉じても使えるようにする
    for op in _ORIENT.get(orientation, []):
        img = img.transpose(op)
    return img


# ------------------------------------------------------------------ ピント
def _laplacian(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)   # センサーノイズの影響を抑える
    return cv2.Laplacian(g, cv2.CV_32F)


def _sharpness(lap: np.ndarray) -> tuple[float, float]:
    global_var = float(lap.var())
    h, w = lap.shape
    n = 8
    tiles = []
    for i in range(n):
        for j in range(n):
            t = lap[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n]
            tiles.append(float(t.var()))
    tiles.sort(reverse=True)
    # 背景ボケの写真でも被写体が合っていればOKにするため、8x8分割の上位4タイルの平均を使う
    return float(np.mean(tiles[:4])), global_var


CELL = 32  # 人物/背景の比較に使う区画の大きさ（1024px 画像上）


def _cell_map(lap: np.ndarray) -> np.ndarray:
    """CELL×CELL 区画ごとの Laplacian 分散"""
    h, w = lap.shape
    ch, cw = h // CELL, w // CELL
    sq = lap[:ch * CELL, :cw * CELL] ** 2
    mean = lap[:ch * CELL, :cw * CELL].reshape(ch, CELL, cw, CELL).mean(axis=(1, 3))
    msq = sq.reshape(ch, CELL, cw, CELL).mean(axis=(1, 3))
    return msq - mean ** 2


def _region_sharpness(cells: np.ndarray, mask: np.ndarray) -> float:
    """マスク内の区画のうち、くっきりした上位 25% の平均（模様の少ない区画の影響を除く）"""
    vals = cells[mask]
    if vals.size < 2:
        return -1.0
    vals = np.sort(vals)[::-1]
    k = max(2, int(np.ceil(vals.size * 0.25)))
    return float(vals[:k].mean())


FACE_MIN_PX = 100   # これより小さい顔（解析画像上）はピントを判定しない（細部が足りず当てにならない）


def _face_sharpness(gray_detail: np.ndarray, face: "people.Box") -> float:
    """顔を幅 200px にそろえて鮮鋭度を測る（顔の大きさによらず比べられるように）。"""
    return _face_measures(gray_detail, face)[0]


def _face_measures(gray_detail: np.ndarray, face: "people.Box") -> tuple[float, float]:
    """(鮮鋭度, 明暗差に対する鮮鋭度)。測れない顔は (-1, -1)。"""
    H, W = gray_detail.shape
    x, y, w, h = face.px(W, H)
    if w < FACE_MIN_PX:
        return -1.0, -1.0
    crop = gray_detail[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return -1.0
    if w > 200:
        crop = cv2.resize(crop, (200, max(1, int(crop.shape[0] * 200 / w))), interpolation=cv2.INTER_AREA)
    sharp = float(_laplacian(crop).var())
    rel = sharp / (float(crop.astype(np.float32).var()) + 25.0) * 1000.0
    return sharp, rel


# ------------------------------------------------------------------ 手ブレ
# 周波数解析による手ブレ検出
#
# 画像を小さな区画に分けてフーリエ変換し、方向ごとに
#   「細かい模様（高い周波数）の量 ÷ 粗い模様（低い周波数）の量」
# を求める。
#  - 手ブレ: ブレた方向だけ細かい模様が消えるので、その方向だけ比が大きく下がる
#  - ピンボケ: 全方向が同じように下がるので、方向による差は出ない
#  - 建物の縦線や水平線など被写体そのものの向き: 細かい模様と粗い模様の
#    両方に同じように現れるので、比をとると打ち消される
# 方向ごとの集計は、各周波数を近い角度へなめらかに振り分ける（ガウス重み）ことで
# 角度による標本数の偏りをなくしている。
# ブレの長さによって効く周波数が変わるので、1/2 に縮小した画像でも同じ計算を行う。
_FT = 64
_FT_WIN = np.outer(np.hanning(_FT), np.hanning(_FT)).astype(np.float32)
_fy, _fx = np.meshgrid(np.fft.fftfreq(_FT), np.fft.fftfreq(_FT), indexing="ij")
_FT_RAD = np.hypot(_fx, _fy)
_FT_PHI = np.degrees(np.arctan2(_fy, _fx)) % 180
_FT_LOW = (_FT_RAD >= 0.03) & (_FT_RAD < 0.08)
_FT_HIGH = (_FT_RAD >= 0.10) & (_FT_RAD < 0.20)
_FT_ANGLES = np.arange(0, 180, 5.0)


def _angle_weights(mask: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    d = np.abs(_FT_PHI[mask][None, :] - _FT_ANGLES[:, None]) % 180
    d = np.minimum(d, 180 - d)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    return w / w.sum(axis=1, keepdims=True)


_FT_WLOW = _angle_weights(_FT_LOW)
_FT_WHIGH = _angle_weights(_FT_HIGH)

# 高速化: 実数の画像のスペクトルは点対称なので、半分（rfft2 の結果）だけで同じ合計が求まる。
# rfft2 の列 0 と列 _FT/2 は 1 回、それ以外の列は（対になる負の周波数の分も含めて）2 回数える。
_FT_HALF = _FT // 2 + 1
_FT_HCOL = np.full(_FT_HALF, 2.0)
_FT_HCOL[0] = _FT_HCOL[-1] = 1.0


def _half_weights(mask_full: np.ndarray, w_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全面の (マスク, 角度の重み) を、rfft2 の半面用に変換する"""
    idx_full = np.full(mask_full.shape, -1)
    idx_full[mask_full] = np.arange(mask_full.sum())
    half_idx = idx_full[:, :_FT_HALF]
    sel = half_idx >= 0
    cols = np.broadcast_to(np.arange(_FT_HALF), half_idx.shape)[sel]
    return sel, w_full[:, half_idx[sel]] * _FT_HCOL[cols][None, :]


_FT_LOW_H, _FT_WLOW_H = _half_weights(_FT_LOW, _FT_WLOW)
_FT_HIGH_H, _FT_WHIGH_H = _half_weights(_FT_HIGH, _FT_WHIGH)
_FT_LOW_HCOUNT = _FT_HCOL[np.broadcast_to(np.arange(_FT_HALF), _FT_LOW_H.shape)[_FT_LOW_H]]
MOTION_PERP_MIN = -5.2   # ブレと直角の方向にこれ以上の細かさが残っていなければ手ブレとみなさない（強いピンボケ対策）


def _dir_profile(ps: np.ndarray) -> np.ndarray:
    """方向ごとの「細かい模様 ÷ 粗い模様」(対数)。ps は rfft2 の半面のパワースペクトル"""
    return np.log((_FT_WHIGH_H @ ps[_FT_HIGH_H] + 1e-12) / (_FT_WLOW_H @ ps[_FT_LOW_H] + 1e-12))


def _motion_dip(g: np.ndarray) -> tuple[float, float, float]:
    """戻り値: (方向による落ち込みの大きさ, その方向[度], 直角方向の細かさ)

    カメラの手ブレは画面全体に同じ向きで出るので、画面を 4 分割した
    それぞれで同じ向きに落ち込みがあるかを確かめ、その中央値を使う。
    （一部だけに写った動く被写体や、一部だけシャープな写真を手ブレと誤判定しないため）"""
    h, w = g.shape
    step = _FT // 2
    ys = np.arange(0, h - _FT + 1, step)
    xs = np.arange(0, w - _FT + 1, step)
    # 区画をまとめて切り出し、まとめてフーリエ変換する（1 つずつ処理するより大幅に速い）
    win = np.lib.stride_tricks.sliding_window_view(g, (_FT, _FT))[::step, ::step][:len(ys), :len(xs)]
    tiles = win.reshape(-1, _FT, _FT)
    sd = tiles.std(axis=(1, 2))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    quad = ((yy.ravel() + _FT / 2 >= h / 2) * 2 + (xx.ravel() + _FT / 2 >= w / 2)).astype(int)
    spectra = []
    for q in range(4):
        idx = np.flatnonzero(quad == q)
        if idx.size == 0:
            spectra.append(np.zeros((_FT, _FT_HALF)))
            continue
        idx = idx[np.argsort(-sd[idx], kind="stable")][: max(2, idx.size // 2)]   # 模様のある上位半分の区画だけ使う
        t = tiles[idx].astype(np.float32)
        t = (t - t.mean(axis=(1, 2), keepdims=True)) * _FT_WIN
        f = np.fft.rfft2(t)
        pw = f.real ** 2 + f.imag ** 2
        low = pw[:, _FT_LOW_H] @ _FT_LOW_HCOUNT          # 全面で合計したのと同じ値
        spectra.append((pw / (low[:, None, None] + 1e-9)).sum(axis=0))   # 区画ごとに正規化して合計
    r = _dir_profile(sum(spectra))
    i = int(np.argmin(r))
    perp = float(r[(i + len(r) // 2) % len(r)])
    # 4 分割それぞれで、全体と同じ向き i にどれだけ落ち込んでいるか
    quad_dips = []
    for ps in spectra:
        if ps.any():
            rq = _dir_profile(ps)
            quad_dips.append(float(np.median(rq) - rq[i]))
    dip = float(np.median(quad_dips)) if quad_dips else 0.0
    # 周波数の向き = 模様が変化する向き。細部が消えた周波数の向き = ブレの方向
    return dip, float(_FT_ANGLES[i]), perp


def _motion(gray: np.ndarray) -> tuple[float, float, float]:
    """戻り値: (手ブレらしさ 0〜1, ブレの方向[度, 0=横 90=縦], 直角方向の細かさ)"""
    g = gray.astype(np.float32)
    best = (0.0, 0.0, -99.0)
    for _ in range(2):            # 等倍 と 1/2（長いブレ用）
        if min(g.shape) < 2 * _FT:
            break
        dip, angle, perp = _motion_dip(g)
        if perp >= MOTION_PERP_MIN and dip > best[0]:
            best = (dip, angle, perp)
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return min(1.0, best[0]), best[1], best[2]


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    v = 0
    for bit in diff.flatten():
        v = (v << 1) | int(bit)
    return v


# ------------------------------------------------------------------ 目
EYE_MIN_FACE_SHARPNESS = 18.0


def _eyes(gray_detail: np.ndarray, faces: list) -> tuple[int, int]:
    eye_c = _eye_cascade()
    if eye_c.empty():
        return 0, 0
    H, W = gray_detail.shape
    checked = missing = 0
    for f in faces:
        x, y, w, h = f.px(W, H)
        if w < 120 or f.score < 0.9:     # 小さい顔（目が 15px 程度しかない）・自信のない顔は判定しない
            continue
        # 正面を向いた顔だけ（両目の間隔が顔幅の 35% 以上で、目が顔の左右ほぼ対称の位置にある）
        rx, _, lx, _ = f.landmarks
        eye_dist = abs(lx - rx) * W
        mid = (lx + rx) / 2 * W
        if eye_dist < w * 0.35 or abs(mid - (x + w / 2)) > w * 0.12:
            continue
        # ボケている顔は目の形が読み取れないので判定しない
        if _face_sharpness(gray_detail, f) < EYE_MIN_FACE_SHARPNESS:
            continue
        # 顔検出が返す目の位置のまわりを、片目ずつ切り出して調べる
        rx, ry, lx, ly = f.landmarks
        half = max(10, int(w * 0.18))
        found = 0
        for ex, ey in ((rx * W, ry * H), (lx * W, ly * H)):
            ex, ey = int(ex), int(ey)
            roi = gray_detail[max(0, ey - half): ey + half, max(0, ex - half): ex + half]
            if roi.size == 0:
                continue
            roi = cv2.equalizeHist(roi)
            eyes = eye_c.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=3,
                                          minSize=(half // 2, half // 2))
            found += len(eyes) > 0
        checked += 1
        if found == 0:      # 両目とも見つからないときだけ「目つぶりの可能性」
            missing += 1
    return checked, missing


# ------------------------------------------------------------------ 解析本体
def analyze(path: str, want_people: bool = True) -> tuple[Metrics, Image.Image | None]:
    """1枚を解析する。どんなファイルでも例外を外に出さず、失敗は Metrics.error で返す。"""
    m = Metrics(path=path)
    try:
        m.mtime = os.path.getmtime(path)
        with Image.open(path) as probe:
            w, h = probe.size
            try:
                if probe.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # 縦向き撮影
                    w, h = h, w
            except Exception:
                pass  # 壊れた EXIF は無視する
            m.width, m.height = w, h
        # 解析用は 1024px 以上になる倍率で縮小デコードする（1/4 などで読むと、読み込み・縮小とも速い）
        full = load_image(path, ANALYSIS_SIZE, fit=False)
    except MemoryError:
        m.ok, m.error = False, "読み込めません: 画像が大きすぎてメモリが足りません"
        return m, None
    except Exception as e:  # 壊れたファイル・非対応形式・クラウド上のみのファイルなど
        m.ok, m.error = False, f"読み込めません: {e}"
        return m, None

    if min(full.size) < 16:
        m.ok, m.error = False, "画像が小さすぎて判定できません"
        return m, None

    try:
        img = full.copy()
        # 整数分の 1 の縮小を先に行うと速い（鮮鋭度の差は 4% 程度で、判定には影響しない）
        img.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.Resampling.LANCZOS, reducing_gap=1.4)
        rgb = np.ascontiguousarray(np.asarray(img))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = _laplacian(gray)
        m.sharpness, m.sharpness_global = _sharpness(lap)
        m.motion_score, m.motion_angle, m.motion_perp = _motion(gray)
        m.brightness = float(gray.mean())
        m.contrast = float(gray.std())
        clipped = gray >= 250
        m.highlight_clip = float(clipped.mean())
        m.highlight_clip_lower = float(clipped[int(clipped.shape[0] * 0.4):].mean())
        m.shadow_clip = float((gray <= 6).mean())
        m.dhash = _dhash(gray)
    except Exception as e:
        m.ok, m.error = False, f"解析できません: {e}"
        return m, None

    if want_people:
        try:
            _analyze_objects(m, rgb, gray, lap, full)
        except Exception:
            pass  # 人物の判定に失敗しても、ほかの判定は続ける

    try:
        thumb = img.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    except Exception:
        thumb = None
    return m, thumb


FACE_ALONE_SCORE = 0.9
FACE_DETECT_SIZE = 640
FACE_REFINE_PX = 48     # 1024px 画像でこれ以上の顔は、周りを切り出して 1024px で検出し直す（目の位置を正確にする）


def _refine_face(rgb: np.ndarray, f):
    """640px で見つけた顔を、1024px 画像の顔の周りだけ切り出して検出し直す。
    目つぶり判定は目の位置のわずかなずれに敏感なので、大きな顔だけ精度を上げる（小さな切り出しなので速い）。"""
    H, W = rgb.shape[:2]
    x, y, w, h = f.px(W, H)
    x0, y0 = max(0, x - w), max(0, y - h)
    x1, y1 = min(W, x + 2 * w), min(H, y + 2 * h)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return f
    cands = people.detect_faces(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not cands:
        return f
    cw, ch = x1 - x0, y1 - y0
    cx, cy = (x + w / 2 - x0) / cw, (y + h / 2 - y0) / ch
    g = min(cands, key=lambda c: (c.x + c.w / 2 - cx) ** 2 + (c.y + c.h / 2 - cy) ** 2)
    sx, sy = cw / W, ch / H
    lm = (g.landmarks[0] * sx + x0 / W, g.landmarks[1] * sy + y0 / H,
          g.landmarks[2] * sx + x0 / W, g.landmarks[3] * sy + y0 / H)
    return people.Box(g.x * sx + x0 / W, g.y * sy + y0 / H, g.w * sx, g.h * sy, g.score, lm)


# ------------------------------------------------------------------ 被写体（物体）
SUBJECT_MIN_AREA = 0.01      # 写っている形がこれ以上の面積の物を被写体の候補にする
SUBJECT_MIN_HEIGHT = 0.25    # または、枠の高さが写真の 25% 以上ある物（細長い人物など）
TAG_MIN_SCORE = 0.40         # この確信度以上で見つかった物をタグにする
MOTION_OBJECTS = 3           # 手ブレを測る物の数（大きい順）


def _face_in(face, obj) -> bool:
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    return obj.x <= cx <= obj.x + obj.w and obj.y <= cy <= obj.y + obj.h


def _crop_motion(gray: np.ndarray, o) -> float:
    """物の枠（少し広げる）の中だけで手ブレを測る。小さすぎれば -1"""
    H, W = gray.shape
    mx, my = o.w * 0.1, o.h * 0.1
    crop = gray[max(0, int((o.y - my) * H)):min(H, int((o.y + o.h + my) * H)),
                max(0, int((o.x - mx) * W)):min(W, int((o.x + o.w + mx) * W))]
    if min(crop.shape) < 2 * _FT:
        return -1.0
    return _motion(crop)[0]


def _analyze_objects(m: Metrics, rgb: np.ndarray, gray: np.ndarray, lap: np.ndarray, full: Image.Image):
    objs = people.detect_objects(rgb)
    persons = [o for o in objs if o.cls == people.PERSON]

    # 顔検出は 640px で行う。判定に使う大きさ（1024px で 50px 以上）の顔は 1024px と同じように見つかり、
    # 処理は半分以下で済む（写真での比較で見落としなし、目の位置のずれは 2048px 換算で約 3px）
    face_src = rgb
    if max(rgb.shape[:2]) > FACE_DETECT_SIZE:
        k = FACE_DETECT_SIZE / max(rgb.shape[:2])
        face_src = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    faces = [f for f in people.detect_faces(cv2.cvtColor(face_src, cv2.COLOR_RGB2BGR))
             if f.w * rgb.shape[1] >= 16]
    faces = [_refine_face(rgb, f) if f.w * rgb.shape[1] >= FACE_REFINE_PX else f for f in faces]
    # 顔だけの検出は誤検出（器や模様が顔に見える）があるので、
    # 人物の中にある顔か、確信度の高い顔だけを残す
    faces = [f for f in faces if f.score >= FACE_ALONE_SCORE or any(_face_in(f, p) for p in persons)]
    m.objects, m.faces = objs, faces
    # 顔のピント・目の判定には 2048px の画像を使う。判定できる大きさの顔があるときだけ作る（作るのは重い）
    gray_detail = None
    if any(f.w * DETAIL_SIZE >= FACE_MIN_PX for f in faces):
        try:
            d = load_image(m.path, DETAIL_SIZE)      # 大きな顔があるときだけ、2048px で読み直す
            gray_detail = cv2.cvtColor(np.asarray(d), cv2.COLOR_RGB2GRAY)
        except Exception:
            gray_detail = None

    # 物ごとに、形の中のピント・背景のピント・白飛びを測る
    cells = _cell_map(lap)
    ch, cw = cells.shape
    for o in objs:
        if o.mask is None:
            continue
        cov = cv2.resize(o.mask[:ch * CELL, :cw * CELL].astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
        # ピンボケの判定は枠（すぐ周りを含む）で測る。黒いウェットスーツのように模様の少ない被写体は
        # 形の中だけだと細部が少なく、ピントが合っていても低く出る。すぐ周り（同じ距離にある水面など）を
        # 含めると安定する（400 枚での比較で誤判定が最も少なかった）
        H, W = lap.shape
        x, y, w, h = o.px(W, H)
        box = np.zeros_like(cov, dtype=bool)
        box[y // CELL:(y + h) // CELL + 1, x // CELL:(x + w) // CELL + 1] = True
        o.sharpness = _region_sharpness(cells, box)
        # 背景との比較には形（輪郭がかかる区画を含む）を使う。枠だと背景が混ざって差が見えにくい
        o.shape_sharpness = _region_sharpness(cells, cov > 0.05)
        # 背景は形から少し離れた区画だけを使う（輪郭のにじみを避ける）
        near = cv2.dilate((cov > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if (~near).sum() >= cells.size * 0.15:
            o.background = _region_sharpness(cells, ~near)
        o.highlight = float((gray[o.mask] >= 250).mean())
        if o.cls == people.PERSON and gray_detail is not None:
            fs = [v for v in (_face_measures(gray_detail, f) for f in faces if _face_in(f, o)) if v[0] >= 0]
            if fs:
                o.face_sharpness, o.face_rel = max(fs)
    # 大きい物だけ、その部分で手ブレを測る（海面などの模様に惑わされないように）
    for o in sorted((o for o in objs if o.mask is not None), key=lambda o: -o.area)[:MOTION_OBJECTS]:
        o.motion = _crop_motion(gray, o)
    for o in objs:
        o.drop_mask()

    if gray_detail is not None:
        m.eyes_checked, m.eyes_missing = _eyes(gray_detail, faces)


@dataclass
class Subject:
    """判定に使う被写体。同じ種類で同じくらい大きく写っている物はまとめて扱う（誰か一人に
    ピントが合っていれば OK）。"""
    cls: int
    objs: list
    sharpness: float
    background: float
    face_sharpness: float
    face_rel: float
    motion: float
    highlight: float
    shape_sharpness: float = -1.0

    @property
    def label(self) -> str:
        return people.LABELS_JA[self.cls]


SUBJECT_AUTO = "auto"


def subject_choices() -> list[tuple[str, str]]:
    """(設定値, 表示名) の一覧。先頭は自動。"""
    return [(SUBJECT_AUTO, "自動（一番大きく写っているもの）")] + \
           [(people.LABELS_EN[i], people.LABELS_JA[i]) for i in range(len(people.LABELS_EN))]


def pick_subject(m: Metrics, mode: str) -> Subject | None:
    cands = [o for o in m.objects if o.sharpness >= 0 and o.score >= TAG_MIN_SCORE
             and (o.area >= SUBJECT_MIN_AREA or o.h >= SUBJECT_MIN_HEIGHT)]
    if not cands:
        return None
    if mode == SUBJECT_AUTO:
        cls = max(cands, key=lambda o: o.area).cls
    else:
        try:
            cls = people.LABELS_EN.index(mode)
        except ValueError:
            return None
    group = [o for o in cands if o.cls == cls]
    if not group:
        return None
    top = max(o.area for o in group)
    group = [o for o in group if o.area >= top * 0.3]
    best = max(group, key=lambda o: o.sharpness)            # いちばんピントが合っているもの
    biggest = max(group, key=lambda o: o.area)
    faces = [(o.face_sharpness, o.face_rel) for o in group if o.face_sharpness >= 0]
    face, face_rel = max(faces) if faces else (-1.0, -1.0)
    return Subject(cls, group, best.sharpness, best.background, face, face_rel, biggest.motion, biggest.highlight,
                   best.shape_sharpness)


def subject_sharpness(m: Metrics, mode: str) -> float:
    s = pick_subject(m, mode)
    return s.sharpness if s else -1.0


MOTION_NO_SUBJECT_EXTRA = 0.15   # 人物で確かめられない写真は手ブレのしきい値をこれだけ上げる
FACE_REL_MIN = 6.0          # 顔の「明暗差に対する鮮鋭度」がこれ以上ならボケとしない


def subject_medians(metrics, mode: str, min_count: int = 5) -> dict[int, float]:
    """同時に読み込んだ写真の被写体の鮮鋭度の中央値を、被写体の種類ごとに求める。
    （自動モードではサーフボードと人物のように種類が混ざる。模様の量が違うので種類ごとに比べる）"""
    by: dict[int, list[float]] = {}
    for m in metrics:
        if m.ok:
            sub = pick_subject(m, mode)
            if sub is not None:
                by.setdefault(sub.cls, []).append(sub.sharpness)
    return {c: float(np.median(v)) for c, v in by.items() if len(v) >= min_count}


def judge(m: Metrics, t: Thresholds, batch_median_sharpness: float | None = None,
          batch_median_subject: dict[int, float] | None = None) -> Verdict:
    """batch_median_subject は subject_medians() の結果（被写体の種類ごとの鮮鋭度の中央値）"""
    v = Verdict()
    if not m.ok:
        v.add("error")
        return v
    sub = pick_subject(m, t.subject) if t.check_subject else None
    exposure_bad = (m.brightness < t.dark or m.brightness > t.bright or m.contrast < t.low_contrast)

    # --- 手ブレ（画面全体が同じ方向に流れている）
    # 人物が写っていれば、人物の部分にも同じブレが出ているときだけ手ブレとする
    # （海面・水平線など、被写体自体に向きのある模様を手ブレと取り違えないため）
    # 人物で確かめられない写真（風景・海・空など）は、模様の向きと取り違えやすいので基準を厳しくする
    if sub is not None and sub.motion >= 0:
        shake_score, shake_thr = min(m.motion_score, sub.motion), t.motion
    else:
        shake_score, shake_thr = m.motion_score, t.motion + MOTION_NO_SUBJECT_EXTRA
    shake = t.check_motion and not exposure_bad and shake_score > shake_thr
    # ほとんど何も写っていない（空や海だけ）写真は向きを測れないので、手ブレではなくピンボケ扱い
    if shake and m.sharpness < t.blur * 0.25:
        shake = False

    # --- ピント
    if sub is not None:
        reasons = []
        name = sub.label
        face_sharp = sub.face_sharpness >= t.face_blur * 2
        # 顔は「細部が少ない」かつ「明暗差に対しても細部が少ない」ときだけボケとする
        # （逆光で暗い顔は細部の量が小さく出るが、明暗差に対しては十分くっきりしている）
        if 0 <= sub.face_sharpness < t.face_blur and sub.face_rel < FACE_REL_MIN:
            reasons.append("顔がボケている")
        elif sub.face_sharpness < 0 and sub.sharpness < t.blur:
            reasons.append(f"{name}がボケている")
        elif (t.relative_blur and batch_median_subject and sub.cls in batch_median_subject
              and sub.sharpness < batch_median_subject[sub.cls] * t.relative_ratio and not face_sharp):
            # 同時に読み込んだ写真の被写体と比べて、この写真の被写体だけが極端にボケている
            reasons.append(f"ほかの写真の{name}よりボケている")
        shape = sub.shape_sharpness if sub.shape_sharpness >= 0 else sub.sharpness
        if (sub.background > 0 and shape < sub.background * t.person_bg_ratio and not face_sharp):
            reasons.append("背景にピントが合っている")
        if reasons and not exposure_bad and not shake:
            v.add("subject_blur", f"{name}: " + "・".join(dict.fromkeys(reasons)))
    else:
        blurry = m.sharpness < t.blur
        if (not blurry and t.relative_blur and batch_median_sharpness
                and m.sharpness < batch_median_sharpness * t.relative_ratio):
            blurry = True
        # 暗い・眠い写真は鮮鋭度も低く出るので、その場合は露出側の理由だけを出す
        if blurry and not exposure_bad and not shake:
            v.add("blur")

    if shake:
        v.add("motion", f"{_angle_name(m.motion_angle)}方向")

    # --- 露出
    if m.brightness < t.dark:
        v.add("dark")
    elif m.shadow_clip > t.shadow_clip:
        v.add("shadow")
    if m.brightness > t.bright:
        v.add("bright")
    elif _highlight_bad(m, t, sub):
        v.add("highlight")
    if m.contrast < t.low_contrast and not (m.brightness < t.dark or m.brightness > t.bright):
        v.add("lowcon")
    if t.check_eyes and m.eyes_checked > 0 and m.eyes_missing > 0:
        v.add("eyes", f"{m.eyes_missing}/{m.eyes_checked}人")
    return v


HIGHLIGHT_ALL = 0.60   # 写真のこれ以上が真っ白なら、どこであっても白飛びの失敗


def _highlight_bad(m: Metrics, t: Thresholds, sub: "Subject | None") -> bool:
    """白飛びを「大事な部分」で判定する。
    被写体が写っていれば被写体の部分、いなければ写真の下 6 割（白く飛んだ空はよくあるので除く）を見る。"""
    if m.highlight_clip > HIGHLIGHT_ALL:
        return True
    if sub is not None and sub.highlight >= 0:
        return sub.highlight > t.highlight_clip
    return m.highlight_clip_lower > t.highlight_clip


def _angle_name(a: float) -> str:
    a = a % 180
    if a < 22.5 or a >= 157.5:
        return "横"
    if a < 67.5:
        return "右下がり斜め"
    if a < 112.5:
        return "縦"
    return "右上がり斜め"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similar_losers(metrics: list[Metrics], max_distance: int) -> dict[str, str]:
    """撮影順に並べて連続する似た写真をグループ化し、
    グループ内で最も鮮鋭な1枚以外を返す {path: ベストのファイル名}。"""
    items = sorted((m for m in metrics if m.ok), key=lambda m: (m.mtime, m.path))
    groups: list[list[Metrics]] = []
    for m in items:
        if groups and hamming(groups[-1][-1].dhash, m.dhash) <= max_distance:
            groups[-1].append(m)
        else:
            groups.append([m])
    out: dict[str, str] = {}
    for g in groups:
        if len(g) < 2:
            continue
        best = max(g, key=lambda m: m.sharpness)
        for m in g:
            if m is not best:
                out[m.path] = Path(best.path).name
    return out


def list_images(paths: list[str], recursive: bool = False) -> list[str]:
    out: list[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            it = pp.rglob("*") if recursive else pp.iterdir()
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT and not _in_output_dir(f):
                    out.append(str(f))
        elif pp.is_file() and pp.suffix.lower() in SUPPORTED_EXT:
            out.append(str(pp))
    seen = set()
    uniq = []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


OUTPUT_DIR_NAME = "_失敗写真"
OK_DIR_NAME = "_OK写真"

# 写真と一緒に動かすファイル（同じ名前の RAW と、現像ソフトの編集情報）
COMPANION_EXT = {".raf", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2", ".dng", ".orf",
                 ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl", ".xmp"}


def companion_files(photo: Path, cache: dict | None = None) -> list[Path]:
    """写真と同じフォルダ・同じ名前（拡張子違い）の RAW などのファイル。
    cache を渡すとフォルダの一覧を 1 回だけ読む（何千枚あるフォルダでも速い）。"""
    cache = {} if cache is None else cache
    folder = photo.parent
    if folder not in cache:
        index: dict[str, list[Path]] = {}
        try:
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in COMPANION_EXT:
                        index.setdefault(os.path.splitext(e.name)[0].lower(), []).append(Path(e.path))
        except OSError:
            pass
        cache[folder] = index
    return list(cache[folder].get(photo.stem.lower(), []))


def _in_output_dir(f: Path) -> bool:
    return OUTPUT_DIR_NAME in f.parts or OK_DIR_NAME in f.parts
