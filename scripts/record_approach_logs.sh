#!/usr/bin/env bash

set -u
set -o pipefail

MAX_BAG_SIZE_BYTES=4294967296
PUBLISHER_WAIT_SECONDS=5

TOPICS=(
  "/emo_robot/perception/gesture_detection"
  "/emo_robot/perception/lidar_target"
  "/emo_robot/perception/person_target"
  "/emo_robot/control/approach_cmd_debug"
)
EXPECTED_TYPES=(
  "emo_robot_interfaces/msg/GestureDetection"
  "emo_robot_interfaces/msg/LidarTarget"
  "emo_robot_interfaces/msg/PersonTarget"
  "geometry_msgs/msg/Twist"
)
NODES=(
  "gesture_detector"
  "lidar_target_detector"
  "person_tracker"
  "approach_controller"
)

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
PROJECT_ROOT="$(
  cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1
  pwd
)"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/data/approach_logs"
SOURCE_CONFIG="${PROJECT_ROOT}/src/emo_robot_bringup/config/gesture_approach.yaml"

usage() {
  printf '%s\n' \
    "Usage: $(basename "$0") [OUTPUT_ROOT]" \
    "" \
    "Record the gesture-approach pipeline topics and selected node logs." \
    "Start gesture_approach.launch.py first. Recording continues until Ctrl+C." \
    "" \
    "Default output root:" \
    "  ${DEFAULT_OUTPUT_ROOT}"
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
if ! ros2 pkg prefix rosbag2_storage_mcap >/dev/null 2>&1; then
  echo "ERROR: rosbag2_storage_mcap is not installed." >&2
  exit 1
fi
bringup_prefix="$(ros2 pkg prefix emo_robot_bringup 2>/dev/null || true)"
logger_executable="${bringup_prefix}/lib/emo_robot_bringup/approach_rosout_logger"
if [[ -z "${bringup_prefix}" || ! -x "${logger_executable}" ]]; then
  echo "ERROR: approach_rosout_logger is not installed." >&2
  echo "Rebuild emo_robot_bringup and source install/local_setup.bash." >&2
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

echo "Checking gesture-approach topics..."
deadline=$((SECONDS + PUBLISHER_WAIT_SECONDS))
while :; do
  all_ready=1
  for topic in "${TOPICS[@]}"; do
    count="$(publisher_count "${topic}")"
    if ! [[ "${count}" =~ ^[0-9]+$ ]] || ((count == 0)); then
      all_ready=0
    fi
  done
  if ((all_ready != 0)) || ((SECONDS >= deadline)); then
    break
  fi
  sleep 0.5
done

missing=0
for index in "${!TOPICS[@]}"; do
  topic="${TOPICS[index]}"
  expected_type="${EXPECTED_TYPES[index]}"
  count="$(publisher_count "${topic}")"
  actual_type="$(ros2 topic type "${topic}" 2>/dev/null || true)"
  if ! [[ "${count}" =~ ^[0-9]+$ ]] || ((count == 0)); then
    echo "  ERROR: no publisher found: ${topic}" >&2
    missing=1
  elif [[ "${actual_type}" != "${expected_type}" ]]; then
    echo "  ERROR: ${topic} has type ${actual_type:-unknown}; expected ${expected_type}" >&2
    missing=1
  else
    echo "  OK: ${topic} (${expected_type})"
  fi
done
if ((missing != 0)); then
  echo "Start gesture_approach.launch.py before running this recorder." >&2
  exit 1
fi

node_list="$(ros2 node list 2>/dev/null || true)"
for node in "${NODES[@]}"; do
  if ! grep -Fxq "/${node}" <<<"${node_list}"; then
    echo "ERROR: required node is not running: /${node}" >&2
    exit 1
  fi
done

mkdir -p -- "${OUTPUT_ROOT}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
session_dir="${OUTPUT_ROOT}/approach_${timestamp}"
suffix=1
while [[ -e "${session_dir}" ]]; do
  session_dir="${OUTPUT_ROOT}/approach_${timestamp}_${suffix}"
  suffix=$((suffix + 1))
done
mkdir -p -- "${session_dir}/parameters" "${session_dir}/config"

if [[ -f "${SOURCE_CONFIG}" ]]; then
  cp -- "${SOURCE_CONFIG}" "${session_dir}/config/gesture_approach.yaml"
else
  echo "WARNING: source config not found: ${SOURCE_CONFIG}" >&2
fi

for node in "${NODES[@]}"; do
  if ! ros2 param dump "/${node}" >"${session_dir}/parameters/${node}.yaml"; then
    echo "ERROR: failed to dump parameters for /${node}" >&2
    exit 1
  fi
done

git_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)"
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'git_commit=%s\n' "${git_commit:-unknown}"
  printf '\n[git_status]\n'
  git -C "${PROJECT_ROOT}" status --short 2>/dev/null || true
  printf '\n[topics]\n'
  for index in "${!TOPICS[@]}"; do
    printf '%s %s\n' "${TOPICS[index]}" "${EXPECTED_TYPES[index]}"
  done
  printf '/rosout rcl_interfaces/msg/Log\n'
} >"${session_dir}/session_info.txt"

bag_dir="${session_dir}/bag"
node_log="${session_dir}/nodes.log"
bag_pid=0
logger_pid=0
stop_requested=0

stop_children() {
  if ((bag_pid > 0)) && kill -0 "${bag_pid}" 2>/dev/null; then
    kill -INT "${bag_pid}" 2>/dev/null || true
  fi
  if ((logger_pid > 0)) && kill -0 "${logger_pid}" 2>/dev/null; then
    kill -INT "${logger_pid}" 2>/dev/null || true
  fi
}

trap 'stop_requested=1; stop_children' INT TERM
trap 'stop_children' EXIT

"${logger_executable}" \
  --output-file "${node_log}" &
logger_pid=$!
sleep 0.5
if ! kill -0 "${logger_pid}" 2>/dev/null; then
  wait "${logger_pid}" || true
  echo "ERROR: approach_rosout_logger exited during startup." >&2
  exit 1
fi

record_topics=("${TOPICS[@]}" "/rosout")
ros2 bag record \
  --storage mcap \
  --output "${bag_dir}" \
  --max-bag-size "${MAX_BAG_SIZE_BYTES}" \
  "${record_topics[@]}" &
bag_pid=$!

echo
echo "Recording gesture-approach logs:"
printf '  %s\n' "${record_topics[@]}"
echo "Session: ${session_dir}"
echo "Node text log: ${node_log}"
echo "Storage: MCAP, no compression, 4 GiB split size"
echo "Press Ctrl+C once to stop and finalize both log streams."
echo

unexpected_exit=0
while kill -0 "${bag_pid}" 2>/dev/null &&
  kill -0 "${logger_pid}" 2>/dev/null; do
  sleep 0.2
done
if ((stop_requested == 0)); then
  unexpected_exit=1
  echo "ERROR: a logging process exited unexpectedly; stopping the other." >&2
  stop_children
fi

bag_status=0
logger_status=0
wait "${bag_pid}" || bag_status=$?
wait "${logger_pid}" || logger_status=$?
trap - INT TERM EXIT

if ((stop_requested != 0)); then
  ((bag_status == 130)) && bag_status=0
  ((logger_status == 130)) && logger_status=0
fi

if [[ ! -f "${bag_dir}/metadata.yaml" ]]; then
  echo "ERROR: rosbag did not create metadata.yaml in ${bag_dir}" >&2
  exit 1
fi

{
  printf '\nstopped_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'bag_status=%s\n' "${bag_status}"
  printf 'logger_status=%s\n' "${logger_status}"
} >>"${session_dir}/session_info.txt"

echo
echo "Recording finalized: ${session_dir}"
ros2 bag info "${bag_dir}" | tee "${session_dir}/bag_info.txt"

if ((unexpected_exit != 0 || bag_status != 0 || logger_status != 0)); then
  echo "ERROR: logging ended with a child-process failure." >&2
  exit 1
fi
