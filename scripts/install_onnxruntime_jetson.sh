#!/usr/bin/env bash

set -euo pipefail

ORT_VERSION="1.23.0"
ORT_WHEEL_NAME="onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"
ORT_WHEEL_SHA256="eb64c57f89f8d152e328227e118c9a36537d3cd6e1bbd3ed4781f83238835c74"
ORT_WHEEL_URL="https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/e1e/9e3dc2f4d5551/${ORT_WHEEL_NAME}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${CONDA_PREFIX:-}" ]] || die "Activate the emobot conda environment first."
[[ "$(basename "${CONDA_PREFIX}")" == "emobot" ]] ||
  die "Expected the emobot environment, got: ${CONDA_PREFIX}"
[[ "$(uname -m)" == "aarch64" ]] ||
  die "This wheel is only for Jetson aarch64."

PYTHON_VERSION="$("${CONDA_PREFIX}/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.10" ]] ||
  die "This wheel requires Python 3.10, got ${PYTHON_VERSION}."

if [[ $# -gt 1 ]]; then
  die "Usage: $0 [path/to/${ORT_WHEEL_NAME}]"
fi

if [[ $# -eq 1 ]]; then
  WHEEL_PATH="$(realpath "$1")"
else
  DOWNLOAD_DIR="${HOME}/Downloads/jetson-wheels"
  mkdir -p "${DOWNLOAD_DIR}"
  WHEEL_PATH="${DOWNLOAD_DIR}/${ORT_WHEEL_NAME}"

  echo "Downloading ONNX Runtime GPU ${ORT_VERSION} for JetPack 6 / CUDA 12.6..."
  DOWNLOAD_OK=false
  if command -v curl >/dev/null 2>&1 &&
    curl --fail --location --http1.1 --retry 5 --retry-all-errors \
      --output "${WHEEL_PATH}.part" \
      "${ORT_WHEEL_URL}"; then
    DOWNLOAD_OK=true
  elif command -v wget >/dev/null 2>&1 &&
    wget --tries=5 \
      --output-document="${WHEEL_PATH}.part" \
      "${ORT_WHEEL_URL}"; then
    DOWNLOAD_OK=true
  fi
  [[ "${DOWNLOAD_OK}" == "true" ]] ||
    die "Wheel download failed. Download it on another computer and pass its local path to this script."
  mv "${WHEEL_PATH}.part" "${WHEEL_PATH}"
fi

[[ -f "${WHEEL_PATH}" ]] || die "Wheel not found: ${WHEEL_PATH}"
echo "${ORT_WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum --check --status ||
  die "Wheel SHA256 verification failed: ${WHEEL_PATH}"

echo "Installing ONNX and ONNX Runtime GPU into ${CONDA_PREFIX}..."
"${CONDA_PREFIX}/bin/python" -m pip uninstall -y \
  onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
"${CONDA_PREFIX}/bin/python" -m pip install \
  "onnx>=1.16,<2" \
  "${WHEEL_PATH}"

"${CONDA_PREFIX}/bin/python" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "CUDAExecutionProvider is unavailable; refusing to use CPU fallback."
    )
PY

echo "ONNX Runtime GPU installation verified."
