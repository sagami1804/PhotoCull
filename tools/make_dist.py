"""配布用フォルダを作る（build.bat で dist\\PhotoCull.exe を作ったあとに実行）

    %LOCALAPPDATA%\\PhotoCull\\venv\\Scripts\\python.exe tools\\make_dist.py

できるもの:
    release\\PhotoCull_v1.0\\              … 配布用フォルダ
        PhotoCull.exe
        README.txt                        … 使い方（受け取った人向け）
        licenses\\                         … 使っている部品のライセンス
        source\\PhotoCull_v1.0_source.zip   … ソースコード（AGPL-3.0 のため同梱）
    release\\PhotoCull_v1.0.zip            … 上のフォルダを ZIP にしたもの（送る用）

ファイル名は英数字だけにしている（ほかの PC で ZIP を展開したときの文字化けを防ぐため）。
文章は UTF-8（BOM 付き）・改行 CRLF で書き出す（Windows のメモ帳で正しく開けるように）。
"""
import os
import shutil
import sys
import time
import zipfile

VERSION = "1.0"
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = f"PhotoCull_v{VERSION}"
OUT = os.path.join(ROOT, "release", NAME)
SITE = next(p for p in sys.path if p.endswith("site-packages"))

# ソースコードに入れるファイル（AI モデルの yolo11s-seg.onnx は大きいので入れず、作り方を tools に入れる）
SOURCE_FILES = [
    "main.py", "build.bat", "requirements.txt", "README.md", "LICENSE",
    "photocull/__init__.py", "photocull/analyzer.py", "photocull/app.py", "photocull/people.py",
    "photocull/models/face_detection_yunet_2023mar.onnx",
    "assets/icon.ico",
    "tools/export_yolo11.py", "tools/make_dist.py",
    "packaging/README_dist.txt", "packaging/THIRD_PARTY_NOTICES.txt",
    "packaging/licenses/AGPL-3.0.txt", "packaging/licenses/LGPL-3.0.txt", "packaging/licenses/GPL-3.0.txt",
]

# 同梱するライブラリのライセンス（インストール済みのパッケージから取る）
LIBRARY_LICENSES = {
    "opencv_LICENSE.txt": os.path.join(SITE, "cv2", "LICENSE.txt"),
    "opencv_LICENSE-3RD-PARTY.txt": os.path.join(SITE, "cv2", "LICENSE-3RD-PARTY.txt"),
    "onnxruntime_LICENSE.txt": os.path.join(SITE, "onnxruntime", "LICENSE"),
    "onnxruntime_ThirdPartyNotices.txt": os.path.join(SITE, "onnxruntime", "ThirdPartyNotices.txt"),
    "numpy_LICENSE.txt": next(os.path.join(SITE, d, "licenses", "LICENSE.txt")
                              for d in os.listdir(SITE) if d.startswith("numpy-") and d.endswith(".dist-info")),
    "pillow_LICENSE.txt": next(os.path.join(SITE, d, "licenses", "LICENSE")
                               for d in os.listdir(SITE) if d.startswith("pillow-") and d.endswith(".dist-info")),
    "python_LICENSE.txt": os.path.join(sys.base_prefix, "LICENSE.txt"),
}


def clean_dir(path: str):
    """前回の出力を消す。OneDrive やエクスプローラーが一時的にフォルダをつかんでいて消せないことがあるので、
    中のファイルを消してフォルダは残す（少し待ってやり直す）。"""
    if not os.path.exists(path):
        return
    for attempt in range(5):
        left = []
        for base, dirs, files in os.walk(path, topdown=False):
            for fn in files:
                try:
                    os.remove(os.path.join(base, fn))
                except OSError:
                    left.append(os.path.join(base, fn))
            for d in dirs:
                try:
                    os.rmdir(os.path.join(base, d))
                except OSError:
                    pass   # 空にできていればフォルダが残っていても問題ない
        if not left:
            return
        time.sleep(1.0)
    sys.exit("前回の配布用フォルダのファイルを消せません（開いているファイルがないか確認してください）:\n" + "\n".join(left))


def write_text(dst: str, text: str):
    """UTF-8（BOM 付き）・CRLF で書く"""
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        f.write(text)


def read_text(src: str) -> str:
    with open(src, encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    exe = os.path.join(ROOT, "dist", "PhotoCull.exe")
    if not os.path.exists(exe):
        sys.exit("dist\\PhotoCull.exe がありません。先に build.bat を実行してください。")
    clean_dir(OUT)
    os.makedirs(os.path.join(OUT, "licenses"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "source"), exist_ok=True)

    shutil.copy2(exe, os.path.join(OUT, "PhotoCull.exe"))
    write_text(os.path.join(OUT, "README.txt"), read_text(os.path.join(ROOT, "packaging", "README_dist.txt")))

    lic = os.path.join(OUT, "licenses")
    write_text(os.path.join(lic, "THIRD_PARTY_NOTICES.txt"),
               read_text(os.path.join(ROOT, "packaging", "THIRD_PARTY_NOTICES.txt")))
    for name in ("AGPL-3.0.txt", "LGPL-3.0.txt", "GPL-3.0.txt"):
        write_text(os.path.join(lic, name), read_text(os.path.join(ROOT, "packaging", "licenses", name)))
    for name, src in LIBRARY_LICENSES.items():
        write_text(os.path.join(lic, name), read_text(src))

    src_zip = os.path.join(OUT, "source", f"{NAME}_source.zip")
    with zipfile.ZipFile(src_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in SOURCE_FILES:
            z.write(os.path.join(ROOT, rel), f"{NAME}_source/{rel}")

    final_zip = os.path.join(ROOT, "release", f"{NAME}.zip")
    if os.path.exists(final_zip):
        os.remove(final_zip)
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _, files in os.walk(OUT):
            for fn in files:
                full = os.path.join(base, fn)
                z.write(full, os.path.relpath(full, os.path.dirname(OUT)))

    print("配布用フォルダ:", OUT)
    for base, _, files in os.walk(OUT):
        for fn in sorted(files):
            full = os.path.join(base, fn)
            print(f"  {os.path.relpath(full, OUT):48} {os.path.getsize(full) / 1e6:8.2f} MB")
    print(f"ZIP: {final_zip}  ({os.path.getsize(final_zip) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
