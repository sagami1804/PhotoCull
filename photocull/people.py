"""物体（タグ）・顔の検出

- 物体: YOLO11s-seg (COCO 80 種類, Ultralytics, AGPL-3.0)  https://github.com/ultralytics/ultralytics
        枠だけでなく、写っている形（マスク）も分かる
- 顔  : YuNet (MIT)  https://github.com/opencv/opencv_zoo

物体検出は ONNX Runtime + DirectML でグラフィックボード（GPU）を使う。DirectML は Windows 標準の
GPU 機能なので、NVIDIA / AMD / Intel のどれでも動き、CUDA などの追加インストールは不要。
GPU が使えなければ ONNX Runtime の CPU 版で動かす。
顔検出は OpenCV で、スレッドごとに読み込む（共有すると壊れるため）。
OpenCV は日本語を含むパスを開けないので、モデルは Python で読んでメモリから渡す。
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

SEG_FILE = "yolo11s-seg.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SEG_SIZE = 640
PERSON = 0
MASK_STORE = 96   # 保存しておくマスクの長辺（プレビューの輪郭表示用。小さくしてメモリを節約）
FULL_MASK_MIN = 0.003   # 枠の面積がこれ以上の物だけ実寸マスクを作る

# COCO の 80 種類（モデルの出力順）と日本語名
LABELS_EN = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
LABELS_JA = [
    "人物", "自転車", "車", "バイク", "飛行機", "バス", "電車", "トラック", "ボート・船", "信号機",
    "消火栓", "一時停止標識", "パーキングメーター", "ベンチ", "鳥", "猫", "犬", "馬", "羊", "牛",
    "象", "熊", "シマウマ", "キリン", "リュック", "傘", "ハンドバッグ", "ネクタイ", "スーツケース", "フリスビー",
    "スキー板", "スノーボード", "ボール", "凧", "野球バット", "野球グローブ", "スケートボード", "サーフボード",
    "テニスラケット", "ボトル", "ワイングラス", "カップ", "フォーク", "ナイフ", "スプーン", "器・ボウル", "バナナ", "りんご",
    "サンドイッチ", "オレンジ", "ブロッコリー", "にんじん", "ホットドッグ", "ピザ", "ドーナツ", "ケーキ", "椅子", "ソファ",
    "鉢植え", "ベッド", "テーブル", "トイレ", "テレビ", "ノートパソコン", "マウス", "リモコン", "キーボード",
    "スマートフォン", "電子レンジ", "オーブン", "トースター", "流し台", "冷蔵庫", "本", "時計", "花瓶",
    "はさみ", "ぬいぐるみ", "ドライヤー", "歯ブラシ",
]


def _model_dir() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


_buf_lock = threading.Lock()
_buffers: dict[str, np.ndarray | None] = {}
_tls = threading.local()


def _model_bytes(name: str) -> np.ndarray | None:
    with _buf_lock:
        if name not in _buffers:
            try:
                with open(os.path.join(_model_dir(), name), "rb") as f:
                    _buffers[name] = np.frombuffer(f.read(), np.uint8)
            except OSError:
                _buffers[name] = None
        return _buffers[name]


def available() -> bool:
    return _model_bytes(SEG_FILE) is not None and _model_bytes(YUNET_FILE) is not None


# ------------------------------------------------------------------ 物体検出の実行環境
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - 同梱漏れなど
    ort = None

_ort_lock = threading.Lock()      # セッション作成用
_gpu_run_lock = threading.Lock()  # DirectML は同時実行に対応していないので 1 つずつ流す
_use_gpu = True
_session = None                   # onnxruntime.InferenceSession | False(使えない) | None(未作成)
_backend = ""                     # "gpu" / "cpu" / "none"
_backend_note = ""


def configure(use_gpu: bool):
    """GPU を使うかどうかを切り替える（次の検出から反映）。"""
    global _use_gpu, _session, _backend, _backend_note
    with _ort_lock:
        if use_gpu != _use_gpu:
            _use_gpu = use_gpu
            _session, _backend, _backend_note = None, "", ""


def _ort_session():
    global _session, _backend, _backend_note
    if _session is not None:
        return _session or None
    with _ort_lock:
        if _session is not None:
            return _session or None
        b = _model_bytes(SEG_FILE)
        if ort is None or b is None:
            _session, _backend = False, "none"
            _backend_note = "物体検出モデルを読み込めないため、タグ・被写体の判定はできません"
            return None
        model = b.tobytes()
        if _use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
            try:
                so = ort.SessionOptions()
                so.enable_mem_pattern = False             # DirectML の必須設定
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                s = ort.InferenceSession(model, so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
                if "DmlExecutionProvider" in s.get_providers():
                    _session, _backend, _backend_note = s, "gpu", ""
                    return s
            except Exception as e:
                _backend_note = f"GPU を使えなかったため CPU で処理します（{e}）"
        elif _use_gpu:
            _backend_note = "使える GPU が見つからないため CPU で処理します"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2   # 写真ごとの並列処理と合わせて CPU を使い切る
        _session = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        _backend = "cpu"
        return _session


def backend() -> tuple[str, str]:
    """(処理方法, 補足) を返す。処理方法は "gpu" / "cpu" / "none"。"""
    _ort_session()
    return _backend, _backend_note


def _run(blob: np.ndarray):
    s = _ort_session()
    if s is None:
        return None
    if _backend == "gpu":
        with _gpu_run_lock:
            return s.run(None, {"images": blob})
    return s.run(None, {"images": blob})


def _yunet():
    fd = getattr(_tls, "yunet", None)
    if fd is None:
        b = _model_bytes(YUNET_FILE)
        fd = (cv2.FaceDetectorYN.create("onnx", b, np.array([], np.uint8), (320, 320), 0.75, 0.3, 5000)
              if b is not None else False)
        _tls.yunet = fd
    return fd or None


# ------------------------------------------------------------------ 検出結果
@dataclass
class Box:
    x: float  # 画像サイズに対する割合 0〜1
    y: float
    w: float
    h: float
    score: float
    landmarks: tuple = ()  # 顔のみ: (右目x, 右目y, 左目x, 左目y) の割合

    def px(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (int(self.x * width), int(self.y * height), int(self.w * width), int(self.h * height))


@dataclass
class Obj(Box):
    """検出した物体 1 つ分。マスクは小さく縮めて保存し、解析中だけ実寸のマスクを持つ。"""
    cls: int = 0
    area: float = 0.0                 # 写真全体に対する、写っている形（マスク）の面積の割合
    mask_small: bytes = b""           # 縮小マスク (packbits)
    mask_shape: tuple = (0, 0)
    # 解析で計算する値（analyzer が埋める）
    sharpness: float = -1.0           # 物の枠（すぐ周りを含む）の鮮鋭度。ピンボケの判定に使う
    shape_sharpness: float = -1.0     # 形（輪郭を含む）の鮮鋭度。背景との比較に使う
    background: float = -1.0          # 形のまわり以外（背景）の鮮鋭度
    highlight: float = -1.0           # 形の中の白飛びの割合
    motion: float = -1.0              # 形のまわりだけで測った手ブレらしさ（大きい物だけ）
    face_sharpness: float = -1.0      # 人物の場合: 中にある顔の鮮鋭度
    face_rel: float = -1.0
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)   # 解析中だけ使う実寸マスク

    @property
    def label(self) -> str:
        return LABELS_JA[self.cls]

    def small_mask(self) -> np.ndarray | None:
        if not self.mask_small:
            return None
        h, w = self.mask_shape
        return np.unpackbits(np.frombuffer(self.mask_small, np.uint8))[:h * w].reshape(h, w).astype(bool)

    def drop_mask(self):
        """解析が終わったら実寸マスクを捨てる（保存はしない）"""
        self.mask = None


def detect_objects(rgb: np.ndarray, min_score: float = 0.35, max_det: int = 40) -> list[Obj]:
    """YOLO11-seg で物体を検出する。各物体に実寸（入力画像と同じ大きさ）のマスクを付けて返す。"""
    h, w = rgb.shape[:2]
    r = min(SEG_SIZE / h, SEG_SIZE / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    padded = np.full((SEG_SIZE, SEG_SIZE, 3), 114, np.uint8)
    padded[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    outs = _run(blob)
    if outs is None:
        return []
    pred = outs[0][0].T                       # (8400, 4 + 80 + 32)
    protos = outs[1][0]                       # (32, 160, 160)
    cls_scores = pred[:, 4:84]
    cls_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    keep = scores > min_score
    if not keep.any():
        return []
    pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    rects = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    idx = np.array(cv2.dnn.NMSBoxesBatched(rects.tolist(), scores.tolist(), cls_ids.tolist(),
                                           min_score, 0.5)).flatten()[:max_det]
    if idx.size == 0:
        return []
    # マスク: 係数 × 原型マスク → シグモイド。原型は 640 の 1/4 (160px)
    coef = pred[idx, 84:]
    m = coef @ protos.reshape(32, -1)
    m = (1.0 / (1.0 + np.exp(-m))).reshape(-1, 160, 160)
    ph, pw = max(1, int(round(nh / 4))), max(1, int(round(nw / 4)))
    objs = []
    for k, i in enumerate(idx):
        x0, y0 = max(0.0, rects[i, 0] / r), max(0.0, rects[i, 1] / r)
        x1, y1 = min(float(w), (rects[i, 0] + rects[i, 2]) / r), min(float(h), (rects[i, 1] + rects[i, 3]) / r)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        s = MASK_STORE / max(w, h)
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        bx0, by0, bx1, by1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        if (x1 - x0) * (y1 - y0) >= w * h * FULL_MASK_MIN:
            # 被写体になりうる大きさの物だけ、実寸のマスクを作る（ピントなどを測るため）
            full = cv2.resize(m[k, :ph, :pw], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            box = np.zeros_like(full)
            box[by0:by1, bx0:bx1] = True
            full &= box                         # 枠の外にはみ出たマスクは消す（Ultralytics と同じ）
            if not full.any():
                continue
            area = float(full.mean())
            small = cv2.resize(full.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0
        else:
            # 小さい物はタグとして記録するだけなので、縮小マスクだけ作る
            full = None
            small = cv2.resize(m[k, :ph, :pw], (sw, sh), interpolation=cv2.INTER_LINEAR) > 0.5
            sb = np.zeros_like(small)
            sb[int(by0 * s):int(np.ceil(by1 * s)) + 1, int(bx0 * s):int(np.ceil(bx1 * s)) + 1] = True
            small &= sb
            area = float(small.mean())
        objs.append(Obj(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, float(scores[i]),
                        cls=int(cls_ids[i]), area=area,
                        mask_small=np.packbits(small).tobytes(), mask_shape=small.shape, mask=full))
    return objs


def detect_faces(bgr: np.ndarray) -> list[Box]:
    fd = _yunet()
    if fd is None:
        return []
    h, w = bgr.shape[:2]
    fd.setInputSize((w, h))
    _, faces = fd.detect(bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh = (float(v) for v in f[:4])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lm = (f[4] / w, f[5] / h, f[6] / w, f[7] / h)
        out.append(Box(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, float(f[14]), tuple(float(v) for v in lm)))
    return out
