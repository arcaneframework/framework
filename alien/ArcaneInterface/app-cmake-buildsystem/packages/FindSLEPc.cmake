#
# Find Arpack includes
#
# This module defines
# SLEPC_INCLUDE_DIRS, where to find headers,
# SLEPC_LIBRARIES, the libraries to link against to use boost.
# SLEPC_FOUND If false, do not try to use boost.

if (NOT SLEPC_ROOT)
    set(SLEPC_ROOT $ENV{SLEPC_ROOT})
endif ()

if (SLEPC_ROOT)
    set(_SLEPC_SEARCH_OPTS NO_DEFAULT_PATH)
else ()
    set(_SLEPC_SEARCH_OPTS)
endif ()

if (NOT SLEPC_FOUND)


    find_library(SLEPC_LIBRARY
            NAMES slepc
            HINTS ${SLEPC_ROOT}
            PATH_SUFFIXES lib
            ${_SLEPC_SEARCH_OPTS}
            )
    mark_as_advanced(SLEPC_LIBRARY)

    find_path(SLEPC_INCLUDE_DIR slepc.h
            HINTS ${SLEPC_ROOT}
            PATH_SUFFIXES include
            ${_SLEPC_SEARCH_OPTS}
            )
    mark_as_advanced(SLEPC_INCLUDE_DIR)
endif ()

# pour limiter le mode verbose
set(SLEPC_FIND_QUIETLY ON)

find_package_handle_standard_args(SLEPC
        DEFAULT_MSG
        SLEPC_INCLUDE_DIR
        SLEPC_LIBRARY)

if (SLEPC_FOUND AND NOT TARGET slepc)

    set(SLEPC_INCLUDE_DIRS ${SLEPC_INCLUDE_DIR})

    set(SLEPC_LIBRARIES ${SLEPC_LIBRARY})

    # slepc main

    add_library(slepc_main UNKNOWN IMPORTED)

    set_target_properties(slepc_main PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SLEPC_INCLUDE_DIRS}")

    set_target_properties(slepc_main PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${SLEPC_LIBRARY}")

    add_library(slepc INTERFACE IMPORTED)

    set_property(TARGET slepc APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "slepc_main")

endif ()
  