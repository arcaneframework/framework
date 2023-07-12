if (Arccon_FOUND)
    return()
endif ()

if (NOT ARCCON_EXPORT_TARGET)
    set(ARCCON_EXPORT_TARGET ${ARCCORE_EXPORT_TARGET})
endif ()

# FetchContent use <name>_SOURCE_DIR
SET(ARCCON_CMAKE_COMMANDS ${Arccon_SOURCE_DIR}/Arccon.cmake)
set(ARCCON_MODULE_PATH ${Arccon_SOURCE_DIR}/build-system/Modules)

#list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
#include(${ARCCON_CMAKE_COMMANDS})

