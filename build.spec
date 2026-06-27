# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir：与 main.py 同目录执行
#   pyinstaller main.spec

import os

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None
project = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(project, "main.py")],
    pathex=[project, os.path.join(project, "src")],
    binaries=[],
    datas=[
        (os.path.join(project, "src", "redlotus"), "redlotus"),
        *copy_metadata("genai_prices"),
        *copy_metadata("pydantic_ai_slim"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
