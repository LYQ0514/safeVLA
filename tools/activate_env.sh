#!/usr/bin/env bash
# safeVLA 评测环境入口：source tools/activate_env.sh
# 说明：只影响当前 shell；缓存 / 配置全部留在项目目录内，不写用户 home。
set -u

_SAFEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_SAFELIBERO_CODE="$_SAFEVLA_ROOT/dataset/SafeLIBERO/safelibero"

if [ ! -x "$_SAFEVLA_ROOT/.venv/bin/python" ]; then
    echo "[activate_env] 未找到 .venv，请先按 docs/operations/RUNBOOK.md 建环境" >&2
    return 1 2>/dev/null || exit 1
fi

# 虚拟环境
export VIRTUAL_ENV="$_SAFEVLA_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

# SafeLIBERO 代码（改版 LIBERO，不安装成包，直接走 PYTHONPATH）
export PYTHONPATH="$_SAFELIBERO_CODE${PYTHONPATH:+:$PYTHONPATH}"

# libero 运行时配置：默认会写 ~/.safelibero，这里改到项目内
export LIBERO_CONFIG_PATH="$_SAFEVLA_ROOT/.cache-local/libero"

# 离屏渲染（容器无 X11，用 EGL 走 NVIDIA）
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 缓存 / 状态落地到项目内
export HF_HOME="$_SAFEVLA_ROOT/.cache-local/huggingface"
export TORCH_HOME="$_SAFEVLA_ROOT/.cache-local/torch"
export XDG_CACHE_HOME="$_SAFEVLA_ROOT/.cache-local/cache"
export MPLCONFIGDIR="$_SAFEVLA_ROOT/.cache-local/matplotlib"

# openpi（π0.5 策略子进程）：默认 /opt/assets/.cache/openpi 是 root 只读，
# openpi 启动时会 chmod 该目录 → PermissionError。改指项目内可写副本（已含 paligemma tokenizer）。
export OPENPI_DATA_HOME="$_SAFEVLA_ROOT/.cache-local/openpi"

mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" "$OPENPI_DATA_HOME"

unset _SAFEVLA_ROOT _SAFELIBERO_CODE
