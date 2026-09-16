#!/bin/bash
set -e

# Default values
PORT=""
BOARD=""
BOARD_VARIANT=""

# Usage helper
usage() {
    echo "Usage: $0 -p <esp32|rp2|unix> [-b <BOARD_NAME>] [-v <BOARD_VARIANT>]"
    echo "  -p : Target port (esp32, rp2, or unix)"
    echo "  -b : Board name (e.g., ESP32_GENERIC, RPI_PICO) [Required for esp32/rp2]"
    echo "  -v : Board variant (optional, e.g., SPIRAM or '')"
    exit 1
}

# Parse command line flags
while getopts "p:b:v:" opt; do
    case ${opt} in
        p ) PORT=$OPTARG ;;
        b ) BOARD=$OPTARG ;;
        v ) BOARD_VARIANT=$OPTARG ;;
        * ) usage ;;
    esac
done

# Strip whitespace from variant in case -v ' ' is passed
BOARD_VARIANT=$(echo "$BOARD_VARIANT" | xargs)

if [ -z "$PORT" ]; then
    echo "Error: Port (-p) is required."
    usage
fi

if [ "$PORT" != "unix" ] && [ -z "$BOARD" ]; then
    echo "Error: Board name (-b) is required for '$PORT' port."
    usage
fi

# 1. Base System Dependencies
echo "==> Installing Base Dependencies..."
sudo apt-get update
sudo apt-get install -y git wget make libncurses-dev flex bison gperf python3 \
    python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev \
    dfu-util libusb-1.0-0

# 2. Workspace Setup
echo "==> Setting up workspace..."
ROOT_DIR=$(pwd)

# Clone ulab, and MicroPython directly inside microml/
[ ! -d "micropython-ulab" ] && git clone https://github.com/v923z/micropython-ulab.git
[ ! -d "micropython" ] && git clone https://github.com/micropython/micropython.git

# Set absolute path to microml module to avoid relative path errors inside ports/
USER_MODULES_DIR="${ROOT_DIR}"
MICROPYTHON_DIR="${ROOT_DIR}/micropython"

# Build mpy-cross
cd "${MICROPYTHON_DIR}"
git submodule update --init
make -C mpy-cross

# 3. Toolchain & Build per Port
if [ "$PORT" = "esp32" ]; then
    echo "==> Setting up ESP-IDF Toolchain..."
    if [ ! -d "${ROOT_DIR}/esp-idf" ]; then
        cd "${ROOT_DIR}"
        git clone -b v5.5.1 --recursive https://github.com/espressif/esp-idf.git
    fi
    cd "${ROOT_DIR}/esp-idf"
    git checkout v5.5.1
    git submodule update --init --recursive
    ./install.sh
    source export.sh

    echo "==> Building MicroPython for ESP32..."
    cd "${MICROPYTHON_DIR}/ports/esp32"
    make submodules
    
    BUILD_CMD="make BOARD=${BOARD} USER_C_MODULES=${USER_MODULES_DIR}"
    if [ -n "${BOARD_VARIANT}" ]; then
        BUILD_CMD="${BUILD_CMD} BOARD_VARIANT=${BOARD_VARIANT}"
    fi
    
    echo "Running: ${BUILD_CMD}"
    eval ${BUILD_CMD}

elif [ "$PORT" = "rp2" ]; then
    echo "==> Installing RP2 Toolchain..."
    sudo apt-get install -y build-essential gcc-arm-none-eabi libnewlib-arm-none-eabi

    echo "==> Building MicroPython for RP2..."
    cd "${MICROPYTHON_DIR}/ports/rp2"
    make submodules
    
    BUILD_CMD="make BOARD=${BOARD} USER_C_MODULES=${USER_MODULES_DIR}"
    if [ -n "${BOARD_VARIANT}" ]; then
        BUILD_CMD="${BUILD_CMD} BOARD_VARIANT=${BOARD_VARIANT}"
    fi

    echo "Running: ${BUILD_CMD}"
    eval ${BUILD_CMD}

elif [ "$PORT" = "unix" ]; then
    echo "==> Installing Unix Port Dependencies..."
    sudo apt-get install -y build-essential libreadline-dev libffi-dev

    echo "==> Building MicroPython for Unix Port..."
    cd "${MICROPYTHON_DIR}/ports/unix"
    make submodules
    
    BUILD_CMD="make USER_C_MODULES=${USER_MODULES_DIR}"
    if [ -n "${BOARD}" ]; then
        BUILD_CMD="${BUILD_CMD} BOARD=${BOARD}"
    fi
    if [ -n "${BOARD_VARIANT}" ]; then
        BUILD_CMD="${BUILD_CMD} BOARD_VARIANT=${BOARD_VARIANT}"
    fi

    echo "Running: ${BUILD_CMD}"
    eval ${BUILD_CMD}

else
    echo "Error: Unsupported port '$PORT'. Choose 'esp32', 'rp2', or 'unix'."
    exit 1
fi

echo "==> Build complete! Find your firmware (firmware.uf2 or rp2, .bin for esp32 and .exe for unix) inside /ports/<portname>/build_<board_name_variant>/ directory"
