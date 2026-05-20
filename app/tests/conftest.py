"""让 pytest 在仓库根目录运行时，也能用 ``app.xxx`` 之外的 ``tools.xxx`` 导入风格。

约定：项目的运行入口是 ``app/`` 目录（``python -m main``），所有内部模块都用
``from tools.xxx``、``from common.xxx`` 这种相对 ``app/`` 的路径。本 conftest
把 ``app/`` 加入 ``sys.path``，使得在任意目录下执行 pytest 都能成功导入。
"""

from __future__ import annotations

import os
import sys

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
