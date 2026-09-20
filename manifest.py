# Location: /home/kiit/test_microml/microml/manifest.py

# 1. Load system default manifest (includes _boot.py for mounting flash VFS)
try:
    # Works for CMake ports (RP2040, ESP32, STM32)
    include("$(PORT_DIR)/boards/manifest.py")
except Exception:
    try:
        # Fallback for standard ports
        include("$(PORT_DIR)/manifest.py")
    except Exception:
        pass

# 2. Freeze your custom microml package
package("microml", base_path="src")