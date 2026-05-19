# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir：与 main.py 同目录执行
#   pyinstaller main.spec

import os
import hashlib
import urllib.request

from PyInstaller.utils.hooks import copy_metadata, collect_submodules

block_cipher = None
project = os.path.dirname(os.path.abspath(SPEC))
build_assets = os.path.join(project, ".build_assets")
os.makedirs(build_assets, exist_ok=True)

_CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
_CL100K_CACHE_KEY = hashlib.sha1(_CL100K_URL.encode("utf-8")).hexdigest()
_CL100K_CACHE_FILE = os.path.join(build_assets, _CL100K_CACHE_KEY)

if not os.path.exists(_CL100K_CACHE_FILE):
    with urllib.request.urlopen(_CL100K_URL, timeout=30) as resp:
        data = resp.read()
    with open(_CL100K_CACHE_FILE, "wb") as f:
        f.write(data)

a = Analysis(
    [os.path.join(project, "main.py")],
    pathex=[project, os.path.join(project, "src")],
    binaries=[],
    datas=[
        (os.path.join(project, "src"), "src"),
        *copy_metadata("genai_prices"),
        *copy_metadata("pydantic_ai_slim"),
        *copy_metadata("pydantic_ai"),
        (_CL100K_CACHE_FILE, "tiktoken_cache"),
    ],
    hiddenimports=[
        *collect_submodules("tiktoken_ext"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(project, "src", "hooks", "tiktoken_cache.py")],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Agent",
)
