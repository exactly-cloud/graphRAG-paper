"""模块4：本体推理校验层（OWL + HermiT）。"""
import glob
import os
import sys

import owlready2

# 定位 conda 环境内的 java.exe，供 HermiT 使用
_cands = (
    glob.glob(os.path.join(sys.prefix, "Library", "bin", "java.exe"))
    + glob.glob(os.path.join(sys.prefix, "Library", "lib", "jvm", "bin", "java.exe"))
    + glob.glob(os.path.join(sys.prefix, "bin", "java"))
)
for _p in _cands:
    if os.path.exists(_p):
        owlready2.JAVA_EXE = _p
        break
