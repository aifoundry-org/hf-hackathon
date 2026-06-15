#!/usr/bin/env bash
# Ensure a cmake >= MIN_CMAKE is available and print the directory containing it.
#
# The ggml-et build (-DGGML_ET=ON) uses file(CONFIGURE ...) (CMake 3.18+). Board
# hosts often ship an older system cmake, which fails with
# "file does not recognize sub-command CONFIGURE". This downloads an official
# Kitware cmake into a work dir when the system one is too old.
#
# Logs go to stderr; the ONLY thing written to stdout is the absolute path of the
# directory containing a good cmake, so callers can do:
#   export PATH="$(.github/ci/scripts/ensure_cmake.sh):$PATH"
set -euo pipefail

MIN_MAJOR=3
MIN_MINOR=18
CMAKE_VERSION="${CMAKE_VERSION:-3.31.6}"
INSTALL_ROOT="${CMAKE_INSTALL_ROOT:-${WORK_ROOT:-/tmp}/cmake-dist}"
log() { echo "[ensure_cmake] $*" >&2; }

version_ok() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  local ver major minor
  ver="$("$bin" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)"
  [[ -n "$ver" ]] || return 1
  major="${ver%%.*}"; minor="${ver#*.}"; minor="${minor%%.*}"
  if (( major > MIN_MAJOR )) || { (( major == MIN_MAJOR )) && (( minor >= MIN_MINOR )); }; then
    return 0
  fi
  return 1
}

# 1. System cmake already new enough?
if version_ok cmake; then
  log "system cmake is new enough ($(cmake --version | head -1))"
  dirname "$(command -v cmake)"
  exit 0
fi
log "system cmake missing or older than ${MIN_MAJOR}.${MIN_MINOR}; bootstrapping ${CMAKE_VERSION}"

# 2. Already bootstrapped in the work dir?
dest="${INSTALL_ROOT}/cmake-${CMAKE_VERSION}-linux-x86_64"
if version_ok "${dest}/bin/cmake"; then
  log "using cached cmake at ${dest}/bin"
  echo "${dest}/bin"
  exit 0
fi

# 3. Download the official Kitware binary.
mkdir -p "$INSTALL_ROOT"
url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
log "downloading ${url}"
curl -fsSL -o "${INSTALL_ROOT}/cmake.tgz" "$url"
tar -xzf "${INSTALL_ROOT}/cmake.tgz" -C "$INSTALL_ROOT"
rm -f "${INSTALL_ROOT}/cmake.tgz"
if ! version_ok "${dest}/bin/cmake"; then
  log "ERROR: bootstrapped cmake at ${dest}/bin still not usable"
  exit 1
fi
log "bootstrapped cmake $("${dest}/bin/cmake" --version | head -1)"
echo "${dest}/bin"
