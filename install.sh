#!/bin/bash
# Install both Blender-Moments conda envs.
#
#   blender_moments - rendering and ground-truth meshes (bpy 4.5.9 + numpy<2).
#   bm_colmap       - COLMAP triangulation (hloc fork + Roma matcher).
#
# Detects the local CUDA toolkit version via nvcc and installs the newest
# torch and torchvision built for it into bm_colmap, from the PyTorch index
# for that CUDA version only. The torch-dependent packages then reuse that
# torch, and fused-local-corr is source-built against it so the C++/CUDA ABI
# matches the installed libtorch.
#
# Usage:  bash install.sh
#
# Prereqs:
#  - conda on PATH
#  - CUDA toolkit available (nvcc on PATH)
#  - gcc available for the fused-local-corr source build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_YAML="${SCRIPT_DIR}/blender_moments_env.yaml"
COLMAP_YAML="${SCRIPT_DIR}/bm_colmap_env.yaml"
COLMAP_REQUIREMENTS="${SCRIPT_DIR}/bm_colmap_requirements.txt"

FUSED_LOCAL_CORR_URL="git+https://github.com/Parskatt/fused-local-corr.git"

echo "==[install.sh]== Step 1/5: Prerequisites"

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not on PATH. Install miniconda first." >&2
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not on PATH. Make the CUDA toolkit available" >&2
    echo "  (for example by loading your cluster's CUDA module) before running this script." >&2
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
if [ -z "${CUDA_VERSION}" ]; then
    echo "ERROR: failed to parse CUDA version from nvcc --version" >&2
    exit 1
fi
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d .)"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "  conda:   $(command -v conda)"
echo "  CUDA:    ${CUDA_VERSION} → torch index: ${TORCH_INDEX}"

echo
echo "==[install.sh]== Step 2/5: Create 'blender_moments' env"
conda env create -f "${BLENDER_YAML}"

echo
echo "==[install.sh]== Step 3/5: Create 'bm_colmap' env"
conda env create -f "${COLMAP_YAML}"

echo
echo "==[install.sh]== Step 4/5: torch for CUDA ${CUDA_VERSION}, then hloc and its dependencies"
if ! conda run -n bm_colmap pip install torch torchvision --index-url "${TORCH_INDEX}"; then
    echo "ERROR: no torch build for CUDA ${CUDA_VERSION} at ${TORCH_INDEX}." >&2
    echo "  Use a CUDA toolkit that PyTorch publishes wheels for (see pytorch.org)." >&2
    exit 1
fi
conda run -n bm_colmap pip install -r "${COLMAP_REQUIREMENTS}"

echo
echo "==[install.sh]== Step 5/5: Build fused-local-corr from source"
echo "  (against the env's torch + locally-loaded CUDA; takes a few minutes)"
conda run -n bm_colmap pip install "${FUSED_LOCAL_CORR_URL}" \
    --no-deps --no-build-isolation

conda run -n bm_colmap python -c \
    "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"
echo
echo "==[install.sh]== Done."
echo "  Rendering and meshes: conda activate blender_moments"
echo "  Triangulation:        conda activate bm_colmap"
