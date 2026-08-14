#!/usr/bin/env bash

set -u
set -o pipefail

POINTCLOUD_TOPIC="/aima/hal/sensor/lidar_chest_front/lidar_pointcloud"
IMU_TOPIC="/aima/hal/sensor/lidar_chest_front/imu"
MAX_BAG_SIZE_BYTES=4294967296
PUBLISHER_WAIT_SECONDS=5

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
PROJECT_ROOT="$(
  cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1
  pwd
)"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/data/lidar_bags"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OUTPUT_ROOT]

Record the X2 chest LiDAR PointCloud2 and IMU topics to an MCAP rosbag.
Recording continues until Ctrl+C is pressed.

Default output root:
  ${DEFAULT_OUTPUT_ROOT}
EOF
}

if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

OUTPUT_ROOT="${1:-${DEFAULT_OUTPUT_ROOT}}"

source_setup() {
  local setup_file="$1"
  local description="$2"
  if [[ ! -f "${setup_file}" ]]; then
    echo "ERROR: ${description} setup file not found: ${setup_file}" >&2
    exit 1
  fi
  local source_status
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  source_status=$?
  set -u
  if ((source_status != 0)); then
    echo "ERROR: failed to load ${description}: ${setup_file}" >&2
    exit "${source_status}"
  fi
}

source_setup "/opt/ros/humble/setup.bash" "ROS 2 Humble"
source_setup "${HOME}/aimdk/install/local_setup.bash" "AimDK"
source_setup "${PROJECT_ROOT}/install/local_setup.bash" "emo_robot"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 command is unavailable after loading the environment." >&2
  exit 1
fi

publisher_count() {
  local topic="$1"
  local count
  count="$(
    ros2 topic info "${topic}" 2>/dev/null |
      awk '/Publisher count:/ {print $3; exit}'
  )"
  printf '%s' "${count:-0}"
}

wait_for_publisher() {
  local topic="$1"
  local deadline=$((SECONDS + PUBLISHER_WAIT_SECONDS))
  local count=0

  while ((SECONDS <= deadline)); do
    count="$(publisher_count "${topic}")"
    if [[ "${count}" =~ ^[0-9]+$ ]] && ((count > 0)); then
      return 0
    fi
    sleep 0.5
  done

  echo "ERROR: no publisher found for required topic: ${topic}" >&2
  return 1
}

echo "Checking required LiDAR topics..."
missing_required=0
for topic in "${POINTCLOUD_TOPIC}" "${IMU_TOPIC}"; do
  if wait_for_publisher "${topic}"; then
    echo "  OK: ${topic}"
  else
    missing_required=1
  fi
done
if ((missing_required != 0)); then
  echo "Start the X2 sensor system and verify the topics before recording." >&2
  exit 1
fi

record_topics=("${POINTCLOUD_TOPIC}" "${IMU_TOPIC}")
for optional_topic in "/tf" "/tf_static"; do
  count="$(publisher_count "${optional_topic}")"
  if [[ "${count}" =~ ^[0-9]+$ ]] && ((count > 0)); then
    record_topics+=("${optional_topic}")
    echo "  Optional TF included: ${optional_topic}"
  else
    echo "  Optional TF unavailable, skipped: ${optional_topic}"
  fi
done

mkdir -p -- "${OUTPUT_ROOT}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
bag_dir="${OUTPUT_ROOT}/lidar_${timestamp}"
suffix=1
while [[ -e "${bag_dir}" ]]; do
  bag_dir="${OUTPUT_ROOT}/lidar_${timestamp}_${suffix}"
  suffix=$((suffix + 1))
done

echo
echo "Recording LiDAR data:"
printf '  %s\n' "${record_topics[@]}"
echo "Output: ${bag_dir}"
echo "Storage: MCAP, no compression, 4 GiB split size"
echo "Press Ctrl+C to stop and finalize the bag."
echo

interrupted=0
trap 'interrupted=1' INT

record_status=0
ros2 bag record \
  --storage mcap \
  --output "${bag_dir}" \
  --max-bag-size "${MAX_BAG_SIZE_BYTES}" \
  "${record_topics[@]}" || record_status=$?

trap - INT

if ((interrupted != 0)) && ((record_status == 130)); then
  record_status=0
fi

if [[ ! -f "${bag_dir}/metadata.yaml" ]]; then
  echo "ERROR: rosbag did not create metadata.yaml in ${bag_dir}" >&2
  exit "${record_status:-1}"
fi

echo
echo "Recording finalized: ${bag_dir}"
ros2 bag info "${bag_dir}"

if ((record_status != 0)); then
  echo "WARNING: ros2 bag record exited with status ${record_status}." >&2
fi
exit "${record_status}"
