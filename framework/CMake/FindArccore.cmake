if (Arccore_FOUND)
    return()
endif ()

if (NOT ARCCORE_EXPORT_TARGET)
    # Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des
    # définies dans 'Arccore'.
    set(ARCCORE_EXPORT_TARGET ${FRAMEWORK_EXPORT_NAME})
endif ()

# add directory only once !
# FetchContent use <name>_SOURCE_DIR
if (NOT TARGET arccore_full)
    set(ARCCORE_WANT_TEST OFF)
    add_subdirectory(${Arccore_SOURCE_DIR} arccore)
endif (NOT TARGET arccore_full)
