if (NOT ALIEN_ROOT)
    message(FATAL_ERROR "Variable 'ALIEN_ROOT' is not set")
endif ()

if (NOT ALIEN_EXPORT_TARGET)
    # Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des
    # définies dans 'Arccore'.
    set(ALIEN_EXPORT_TARGET ${FRAMEWORK_EXPORT_NAME})
endif ()

# add directory only once !
if (NOT TARGET Alien::alien_core)
    #add_subdirectory(${ALIEN_ROOT})
    # For LoadAlienTest
    list(APPEND CMAKE_MODULE_PATH ${ALIEN_ROOT}/cmake)
else (NOT TARGET Alien::alien_core)
    list(APPEND CMAKE_MODULE_PATH ${ALIEN_ROOT}/cmake)
endif (NOT TARGET Alien::alien_core)
