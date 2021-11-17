arccon_return_if_package_found(MTL4)

if(NOT MTL4_ROOT)
  set(MTL4_ROOT $ENV{MTL4_ROOT})
endif()

if(MTL4_ROOT)
  set(_MTL4_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_MTL4_SEARCH_OPTS)
endif()


if(NOT MTL4_FOUND)

  find_path(MTL4_INCLUDE_DIR boost/numeric/mtl/mtl.hpp
            HINTS ${MTL4_ROOT}
            PATH_SUFFIXES include
            ${_MTL4_SEARCH_OPTS}
           )
  mark_as_advanced(MTL4_INCLUDE_DIR)

endif()

# pour limiter le mode verbose
set(MTL4_FIND_QUIETLY ON)

find_package_handle_standard_args(MTL4
        DEFAULT_MSG
        MTL4_INCLUDE_DIR)

if(MTL4_FOUND AND NOT TARGET mtl4)

  set(MTL4_INCLUDE_DIRS ${MTL4_INCLUDE_DIR})

  add_library(mtl4 INTERFACE IMPORTED)

  set_target_properties(mtl4 PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${MTL4_INCLUDE_DIRS}")

endif()
