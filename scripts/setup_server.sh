#!/usr/bin/env bash
# Install host dependencies for VideoWala on Ubuntu 24.04 (Noble):
#   Docker (+ Compose plugin), Python 3.11, NVIDIA drivers, ffmpeg/media libs, Node.js, Yarn (Corepack).
#
# Usage (on the server):
#   sudo ./tools/install_ubuntu_server_deps.sh
#
# After a successful NVIDIA driver install, reboot once so the new kernel module loads.
#
# Optional env:
#   VIDEOVALA_SKIP_OS_CHECK=1  — continue even if /etc/os-release is not Ubuntu 24.04
#   VIDEOVALA_NODE_MAJOR=22    — Node major (18, 20, 22, …); default 22

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
NODE_MAJOR="${VIDEOVALA_NODE_MAJOR:-22}"

log() { printf '[%s] %s\n' "$SCRIPT_NAME" "$*"; }
die() { log "error: $*" >&2; exit 1; }

if [[ ${EUID:-0} -ne 0 ]]; then
  die "run as root, e.g. sudo $0"
fi

export DEBIAN_FRONTEND=noninteractive

if [[ -r /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
fi
if [[ "${VIDEOVALA_SKIP_OS_CHECK:-}" != "1" ]]; then
  [[ "${ID:-}" == "ubuntu" ]] || die "expected Ubuntu (set VIDEOVALA_SKIP_OS_CHECK=1 to override)"
  [[ "${VERSION_ID:-}" == "24.04" ]] || die "expected Ubuntu 24.04 (set VIDEOVALA_SKIP_OS_CHECK=1 to override)"
fi

pkg_installed() {
  local p=$1
  dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q '^install ok installed$'
}

apt_update() {
  apt-get update -qq
}

ensure_docker_group() {
  [[ -n "${SUDO_USER:-}" ]] && id "$SUDO_USER" &>/dev/null || return
  if id -nG "$SUDO_USER" | tr ' ' '\n' | grep -qx docker; then
    return
  fi
  usermod -aG docker "$SUDO_USER"
  log "added user '$SUDO_USER' to group 'docker' (log out and back in for it to apply)"
}

install_base_apt() {
  local pkgs=(gnupg software-properties-common apt-transport-https build-essential ubuntu-drivers-common)
  local missing=()
  local p
  for p in "${pkgs[@]}"; do
    pkg_installed "$p" || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "skip base apt packages (already installed)"
    return
  fi
  log "installing base apt packages"
  apt_update
  apt-get install -y --no-install-recommends "${missing[@]}"
}

install_docker() {
  if command -v docker &>/dev/null && docker compose version &>/dev/null; then
    log "skip Docker Engine and Compose plugin (already present)"
    ensure_docker_group
    return
  fi
  log "installing Docker Engine and Compose plugin"
  install -d -m 0755 /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
  fi
  # shellcheck source=/dev/null
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    >/etc/apt/sources.list.d/docker.list
  apt_update
  apt-get install -y --no-install-recommends \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
  systemctl enable --now docker
  ensure_docker_group
}

deadsnakes_configured() {
  grep -rqs 'deadsnakes/ppa' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null
}

install_python311() {
  if command -v python3.11 &>/dev/null \
    && python3.11 -V 2>&1 | grep -q 'Python 3\.11\.' \
    && pkg_installed python3.11 \
    && pkg_installed python3.11-venv \
    && pkg_installed python3.11-dev \
    && python3.11 -m pip --version &>/dev/null; then
    log "skip Python 3.11 (interpreter, venv, dev, and pip already present)"
    return
  fi
  log "installing Python 3.11 (deadsnakes; Noble default is 3.12)"
  if ! deadsnakes_configured; then
    add-apt-repository -y ppa:deadsnakes/ppa
  else
    log "deadsnakes PPA already configured"
  fi
  apt_update
  apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev
  if ! python3.11 -m pip --version &>/dev/null; then
    log "bootstrapping pip for python3.11"
    if ! python3.11 -m ensurepip --upgrade --default-pip &>/dev/null; then
      curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 - --ignore-installed
    fi
  fi
}

install_nvidia() {
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    log "skip NVIDIA driver install (nvidia-smi already works)"
    return
  fi
  log "installing recommended NVIDIA driver (Quadro RTX / desktop class)"
  ubuntu-drivers autoinstall
  apt-get install -y --no-install-recommends nvidia-utils-550 2>/dev/null \
    || apt-get install -y --no-install-recommends nvidia-utils-535 2>/dev/null \
    || true
  log "NVIDIA: reboot when finished so 'nvidia-smi' works."
}

install_media_libs() {
  local pkgs=(ffmpeg libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 libgomp1)
  local missing=()
  local p
  for p in "${pkgs[@]}"; do
    pkg_installed "$p" || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]] && command -v ffmpeg &>/dev/null && command -v ffprobe &>/dev/null; then
    log "skip media libraries (ffmpeg stack already present)"
    return
  fi
  log "installing ffmpeg and common headless OpenCV / ML deps"
  apt_update
  apt-get install -y --no-install-recommends "${pkgs[@]}"
}

install_node() {
  local want_major=$NODE_MAJOR
  local have_major=""
  if command -v node &>/dev/null; then
    have_major=$(node -p "process.versions.node.split('.')[0]" 2>/dev/null || true)
  fi
  if [[ -n "$have_major" && "$have_major" == "$want_major" ]]; then
    log "skip Node.js (already ${want_major}.x)"
    return
  fi
  log "installing Node.js ${want_major}.x (NodeSource)"
  curl -fsSL "https://deb.nodesource.com/setup_${want_major}.x" | bash -
  apt-get install -y --no-install-recommends nodejs
}

install_yarn() {
  command -v node &>/dev/null || {
    log "skip Yarn (Node.js not installed)"
    return
  }
  if command -v yarn &>/dev/null && yarn --version &>/dev/null; then
    log "skip Yarn (already on PATH)"
    return
  fi
  command -v corepack &>/dev/null || die "Node.js should ship corepack; cannot enable Yarn"
  log "installing Yarn via Corepack"
  corepack enable
  corepack prepare yarn@stable --activate
}

main() {
  install_base_apt
  install_docker
  install_python311
  install_nvidia
  install_media_libs
  install_node
  install_yarn
  log "done."
  log "next: reboot if NVIDIA was newly installed, then verify: nvidia-smi, docker run --rm hello-world, python3.11 -V, node -v, yarn -v, ffmpeg -version"
}

main "$@"
