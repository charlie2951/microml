# my_external_modules/custom_math/micropython.cmake

# 1. Create an INTERFACE library for your module
add_library(usermod_custom_math INTERFACE)

# 2. Add your C source files (use CMAKE_CURRENT_LIST_DIR to keep paths absolute)
target_sources(usermod_custom_math INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/microml.c
    ${CMAKE_CURRENT_LIST_DIR}/mlp.c
)

# 3. Specify where the compiler should look for your header files
target_include_directories(usermod_custom_math INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

# 4. Link your module to the master MicroPython user-module target
target_link_libraries(usermod INTERFACE usermod_custom_math)
