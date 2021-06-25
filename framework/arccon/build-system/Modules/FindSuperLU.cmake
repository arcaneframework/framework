include (${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)
arccon_return_if_package_found(SuperLU)

find_library(SuperLU_LIBRARY_DIST
        NAMES superlu_dist)

find_library(SuperLU_LIBRARY_SEQ
        NAMES superlu)

if (SuperLU_LIBRARY_DIST AND SuperLU_LIBRARY_SEQ)
    set(SuperLU_LIBRARIES "${SuperLU_LIBRARY_DIST}" "${SuperLU_LIBRARY_SEQ}")
endif()

# On debian/ubuntu, headers can be found in a /usr/include/"pkg"
find_path(SuperLU_INCLUDE_DIRS slu_cdefs.h
        PATH_SUFFIXES SuperLU superlu)
mark_as_advanced(SuperLU_INCLUDE_DIRS)

set(SuperLU_INCLUDE_DIRS ${SuperLU_INCLUDE_DIR})

message(STATUS "SuperLU_INCLUDE_DIRS=${SuperLU_INCLUDE_DIRS}")
message(STATUS "SuperLU_LIBRARIES=${SuperLU_LIBRARIES}")
message(STATUS "SuperLU_LIBRARY=${SuperLU_LIBRARY}")
message(STATUS "SuperLU_LIBRARY_DIR=${SuperLU_LIBRARY_DIR}")
mark_as_advanced(SuperLU_LIBRARIES)

find_package_handle_standard_args(SuperLU
        DEFAULT_MSG
        SuperLU_INCLUDE_DIRS
        SuperLU_LIBRARIES)

arccon_register_package_library(SuperLU SuperLU)