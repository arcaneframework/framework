include(ExternalProject)

set(BASE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

# Note: in the following, CMAKE_PREFIX_PATH is transmitted to external projects
# using CACHE and not command line. With CMAKE_ARGS, it was truncated, so CMAKE_CACHE_ARGS is MANDATORY !

set(CMAKE_ARGS_ARCCON ${CL_ARGS})
list(APPEND CMAKE_ARGS_ARCCON "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/arccon")

ExternalProject_Add(arccon
        PREFIX modules/arccon
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arccon
        CMAKE_ARGS ${CMAKE_ARGS_ARCCON}
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
        BUILD_ALWAYS
        )

set(CMAKE_ARGS_ARCCORE ${CL_ARGS})
list(APPEND CMAKE_ARGS_ARCCORE "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/arccore"
        "-DArccon_ROOT=${BASE_INSTALL_DIR}/arccon"
        "-DArccore_USE_MPI:BOOL=ON")

ExternalProject_Add(arccore
        PREFIX modules/arccore
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arccore
        CMAKE_ARGS ${CMAKE_ARGS_ARCCORE}
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
                         -DBUILD_SHARED_LIBS:BOOL=true
        BUILD_ALWAYS
        DEPENDS arccon
        )

set(CMAKE_ARGS_ALIEN ${CL_ARGS})
list(APPEND CMAKE_ARGS_ALIEN "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/alien"
        "-DArccon_ROOT=${BASE_INSTALL_DIR}/arccon"
        "-DArccore_ROOT=${BASE_INSTALL_DIR}/arccore")

ExternalProject_Add(alien
        PREFIX modules/alien
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/alien
        CMAKE_ARGS ${CMAKE_ARGS_ALIEN}
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
        BUILD_ALWAYS
        DEPENDS arccon arccore
        )

set(CMAKE_ARGS_PLUGINS ${CL_ARGS})
list(APPEND CMAKE_ARGS_PLUGINS
        "-DArccon_ROOT=${BASE_INSTALL_DIR}/arccon"
        "-DArccore_ROOT=${BASE_INSTALL_DIR}/arccore"
        "-DAlien_ROOT=${BASE_INSTALL_DIR}/alien")

if (ALIEN_PLUGIN_HYPRE)
    ExternalProject_Add(alien_hypre
            PREFIX modules/alien
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugins/hypre
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/hypre" ${CMAKE_ARGS_ALIEN}
            CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
            BUILD_ALWAYS
            DEPENDS arccon arccore alien
            )
endif (ALIEN_PLUGIN_HYPRE)
if (ALIEN_PLUGIN_PETSC)
    ExternalProject_Add(alien_petsc
            PREFIX modules/alien
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugins/petsc
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/petsc" ${CMAKE_ARGS_ALIEN}
            CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
            BUILD_ALWAYS
            DEPENDS arccon arccore alien
            )
endif (ALIEN_PLUGIN_PETSC)
if (ALIEN_PLUGIN_SUPERLU)
    ExternalProject_Add(alien_superlu
            PREFIX modules/alien
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugins/superlu
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/superlu" ${CMAKE_ARGS_ALIEN}
            CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
            BUILD_ALWAYS
            DEPENDS arccon arccore alien
            )
endif (ALIEN_PLUGIN_SUPERLU)
if (ALIEN_PLUGIN_TRILINOS)
    ExternalProject_Add(alien_trilinos
            PREFIX modules/alien
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugins/trilinos
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_INSTALL_DIR}/trilinos" ${CMAKE_ARGS_ALIEN}
            CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${CMAKE_PREFIX_PATH}
            BUILD_ALWAYS
            DEPENDS arccon arccore alien
            )
endif (ALIEN_PLUGIN_TRILINOS)
