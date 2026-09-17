#!/usr/bin/env bash
# Install T-FunS3D on Linux x86_64. CUDA and build tools stay in one Conda env.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${ENV_NAME:-T-FunS3D}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-0}
export MAX_JOBS=${MAX_JOBS:-2}

OPENMASK="$ROOT/third-party/openmask3d"
POINTNET="$OPENMASK/openmask3d/class_agnostic_mask_computation/third_party/pointnet2"
SOURCES="$ROOT/third-party"
SEGMENTATOR="$SOURCES/segmentator"
CURRENT_STAGE="startup"
STAGE_NUMBER=0
TOTAL_STAGES=10
if [[ $INSTALL_FLASH_ATTN == 1 ]]; then
    TOTAL_STAGES=11
fi

usage() {
    cat <<'USAGE'
Usage: bash install.sh

Creates or reuses the T-FunS3D Conda environment and installs its CUDA stack.
Run from any directory. All native source checkouts live under third-party/.

Optional environment variables:
  ENV_NAME=name          Conda environment (default: T-FunS3D)
  MAX_JOBS=number        Parallel compiler jobs (default: 2)
  INSTALL_FLASH_ATTN=1   Also build and test FlashAttention 2 (default: 0)

Use bash install.sh --help to show this message without installing anything.
USAGE
}
if (( $# )); then
    if (( $# == 1 )) && [[ $1 == --help ]]; then
        usage
        exit 0
    fi
    usage >&2
    exit 2
fi

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}
on_exit() {
    local status=$?
    if (( status != 0 )); then
        printf 'Installation stopped in stage "%s" (exit %d).\n' \
            "$CURRENT_STAGE" "$status" >&2
    fi
}
trap on_exit EXIT

run_stage() {
    local label=$1 started=$SECONDS
    shift
    STAGE_NUMBER=$((STAGE_NUMBER + 1))
    CURRENT_STAGE=$label
    printf '\n[%d/%d] Starting: %s\n' "$STAGE_NUMBER" "$TOTAL_STAGES" "$label"
    "$@"
    printf '[%d/%d] Completed: %s (%ds)\n' \
        "$STAGE_NUMBER" "$TOTAL_STAGES" "$label" "$((SECONDS - started))"
}

check_host() {
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Requires Linux x86_64.'
    [[ $ENV_NAME != base && $ENV_NAME =~ ^[a-zA-Z0-9_.-]+$ ]] || fail 'Use a dedicated Conda environment name (not base).'
    [[ $MAX_JOBS =~ ^[1-9][0-9]*$ ]] || fail 'MAX_JOBS must be a positive integer.'
    [[ $INSTALL_FLASH_ATTN == 0 || $INSTALL_FLASH_ATTN == 1 ]] || fail 'INSTALL_FLASH_ATTN must be 0 or 1.'
    for tool in conda git nvidia-smi; do
        command -v "$tool" >/dev/null || fail "Missing $tool. Install Miniforge/Git or an NVIDIA driver first."
    done
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
}

activate_environment() {
    # Noninteractive Bash has no conda function until conda.sh is sourced.
    local conda_base
    conda_base=$(conda info --base)
    # Conda's compiler activation hooks can read unset backup variables.
    set +u
    source "$conda_base/etc/profile.d/conda.sh"
    if conda env list --json | "$conda_base/bin/python" -c \
        'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$ENV_NAME"; then
        conda activate "$ENV_NAME"
        python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing environment must use Python 3.10; choose another ENV_NAME."'
    else
        conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip
        conda activate "$ENV_NAME"
    fi
}

install_toolchain() {
    # The full toolkit supplies nvcc. GCC 11 supports CUDA 12.1 even when the
    # host's default compiler is too new (for example, Ubuntu 24.04's GCC 13).
    conda install -y --override-channels -c nvidia/label/cuda-12.1.1 -c conda-forge \
        cuda-toolkit=12.1.1 gcc_linux-64=11 gxx_linux-64=11 \
        'libblas=*=*openblas' openblas 'cmake>=3.24,<4' ninja make
    set -u
    export CUDA_HOME="$CONDA_PREFIX"
    export CUDA_PATH="$CONDA_PREFIX"
    export CUDACXX="$CONDA_PREFIX/bin/nvcc"
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    export CUDAHOSTCXX="$CXX"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
    export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib"
    export LD_LIBRARY_PATH="$LIBRARY_PATH"
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH
    export PIP_CONSTRAINT="$ROOT/requirements-install.txt"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    "$CUDACXX" --version
    "$CXX" --version
}

install_pytorch() {
    python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0
    python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121
    # Compile for all visible GPUs. The helper rejects architectures too new
    # for the fixed CUDA/PyTorch versions instead of selecting the wrong GPU.
    export TORCH_CUDA_ARCH_LIST
    TORCH_CUDA_ARCH_LIST=$(python "$ROOT/scripts/install_support.py" architectures)
    printf 'CUDA architectures: %s\n' "$TORCH_CUDA_ARCH_LIST"
}

install_python_packages() {
    python -m pip install -r "$ROOT/requirements-install.txt"
    python -m pip install --no-deps --only-binary=:all: torch-scatter==2.1.2 \
        --no-index --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
}

fetch_source() {
    local name=$1 url=$2 revision=$3
    if [[ ! -d $SOURCES/$name ]]; then
        git clone "$url" "$SOURCES/$name"
        git -C "$SOURCES/$name" checkout --detach "$revision"
    fi
    [[ $(git -C "$SOURCES/$name" rev-parse HEAD) == "$revision" ]] || \
        fail "Unexpected revision in $SOURCES/$name; move it aside and retry."
}

prepare_sources() {
    if [[ ! -f $OPENMASK/pyproject.toml ]]; then
        git -C "$ROOT" submodule update --init --recursive -- third-party/openmask3d
    fi
    [[ -f $POINTNET/setup.py ]] || fail "Missing pointnet2 sources in $POINTNET"
    mkdir -p "$SOURCES"
    fetch_source MinkowskiEngine https://github.com/NVIDIA/MinkowskiEngine.git 9f81ae66b33b883cd08ee4f64d08cf633608b118
    fetch_source segmentator https://github.com/Karbo123/segmentator.git 4c6126551685166c6c300551e9ad63db988928c4
    python "$ROOT/scripts/install_support.py" patch "$SOURCES"
}

build_cuda_extensions() {
    # These setup scripts need the already-installed torch and local nvcc.
    python -m pip install --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
    # MinkowskiEngine parses --blas itself. pip --config-settings would not
    # forward these options to setup.py, so build one wheel explicitly.
    (
        cd "$SOURCES/MinkowskiEngine"
        python setup.py bdist_wheel --blas=openblas \
            --blas_include_dirs="$CONDA_PREFIX/include" \
            --blas_library_dirs="$CONDA_PREFIX/lib" --force_cuda
        python -m pip install --no-deps --force-reinstall dist/*.whl
    )
    python -m pip install --no-build-isolation --no-deps "$POINTNET"
}

install_project_packages() {
    python -m pip install --no-build-isolation \
        'git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16' \
        'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
    # Editable installs retain nested OpenMask3D modules and configs.
    python -m pip install --no-build-isolation --no-deps -e "$OPENMASK" -e "$ROOT"
}

build_segmentator() {
    cmake -S "$SEGMENTATOR/csrc" -B "$SEGMENTATOR/csrc/build" \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$CONDA_PREFIX" \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
        -DPYTHON_EXECUTABLE="$(command -v python)" \
        -DPYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
        -DPYTHON_LIBRARY="$(python -c 'import os,sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')" \
        -DCMAKE_INSTALL_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
    cmake --build "$SEGMENTATOR/csrc/build" --parallel "$MAX_JOBS"
    cmake --install "$SEGMENTATOR/csrc/build"
    # Upstream's install(CODE) may leave an old link. Repair it explicitly.
    python - "$SEGMENTATOR" <<'PYLINK'
import os
from pathlib import Path
import sys
import sysconfig

source = Path(sys.argv[1]).resolve()
link = Path(sysconfig.get_path("platlib")) / "segmentator"
if link.exists() and not link.is_symlink():
    raise RuntimeError(f"Refusing to replace a non-link package at {link}")
temp = link.with_name(f".segmentator-link-{os.getpid()}")
temp.symlink_to(source)
temp.replace(link)
PYLINK
}

install_flash_attention() {
    python -c 'import torch; assert all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count())), "FlashAttention 2 requires Ampere, Ada, or Hopper GPUs."'
    FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation flash-attn==2.6.3
    python "$ROOT/scripts/install_support.py" flash
}

validate_installation() {
    python -m pip check
    python "$ROOT/scripts/install_support.py" smoke
}

run_stage 'Host prerequisites' check_host
run_stage 'Conda environment' activate_environment
run_stage 'CUDA toolkit and compilers' install_toolchain
run_stage 'PyTorch and GPU detection' install_pytorch
run_stage 'Python dependencies' install_python_packages
run_stage 'Third-party source preparation' prepare_sources
run_stage 'CUDA extensions' build_cuda_extensions
run_stage 'Project Python packages' install_project_packages
run_stage 'Segmentator' build_segmentator
if [[ $INSTALL_FLASH_ATTN == 1 ]]; then
    run_stage 'FlashAttention (optional)' install_flash_attention
fi
run_stage 'Installation checks' validate_installation
printf '\nInstallation complete. Activate with: conda activate %s\n' "$ENV_NAME"
printf 'Keep %s: segmentator links to its compiled library there.\n' "$SEGMENTATOR"
