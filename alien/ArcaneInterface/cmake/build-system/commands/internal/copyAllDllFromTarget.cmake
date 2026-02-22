function(copyAllDllFromTarget target)

  if(NOT TARGET ${target})
    logFatalError("target ${target} not defined")
  endif()

  string(REPLACE "arccon::" "" target_name ${target})

  set(target ${target_name})

  message(STATUS "copy dll for target ${target}")

  string(TOUPPER ${target} TARGET)

  # Handle target already loaded in another scope (ex when using arcane before alien)
  if ("${${TARGET}_LIBRARIES}" STREQUAL "")
    if (TARGET arcconpkg_${target})
      get_target_property(${TARGET}_LIBRARIES arcconpkg_${target} INTERFACE_LINK_LIBRARIES)
    elseif (TARGET arccon::${target})
      get_target_property(${TARGET}_LIBRARIES arccon::${target} INTERFACE_LINK_LIBRARIES)
    endif ()
  endif ()

  # Handle Arcane libs with Arcane 3: this function wants a list of path prefixed libs while target only contains libs names (arcane_mpi...)
  if (${target} STREQUAL "Arcane::arcane_full")
    unset(${TARGET}_LIBRARIES)
    # Build Arcane library list for dll copy
    get_target_property(${target}_LIBRARIES ${target} INTERFACE_LINK_LIBRARIES)
    foreach (lib ${${target}_LIBRARIES})
      # change Arccane::arcane_lib (mpi, core, ...) into arcane_lib (mpi,core...)
      string(REPLACE "Arcane::" "" lib_name ${lib})
      # get full lib pat : install_path/arcane_lib
      find_library(${lib}_LIBRARIES ${lib_name} HINTS ${ARCANE_PREFIX_DIR}/lib)
      list(APPEND ${TARGET}_LIBRARIES ${${lib}_LIBRARIES})
    endforeach ()
    message(STATUS "Arcane libraries ${${TARGET}_LIBRARIES}")
  endif ()

  # Handle Arccore libs with Arcane 3: same manip than Arcane
  if (${target} STREQUAL "Arccore::arccore_full")
    unset(${TARGET}_LIBRARIES)
    # Build Arccore library list for dll copy
    get_target_property(${target}_LIBRARIES ${target} INTERFACE_LINK_LIBRARIES)
    foreach (lib ${${target}_LIBRARIES})
      string(REPLACE "Arccore::" "" lib_name ${lib})
      find_library(${lib}_LIBRARIES ${lib_name} HINTS ${ARCANE_PREFIX_DIR}/lib)
      list(APPEND ${TARGET}_LIBRARIES ${${lib}_LIBRARIES})
    endforeach ()
    message(STATUS "Arcane libraries ${${TARGET}_LIBRARIES}")
  endif ()
  
  # Handle ArcGeoSim libs with Arcane 3: same manip than Arcane
  if (${target} STREQUAL "ArcGeoSim::arcgeosim" OR ${target} STREQUAL "ArcGeoSim::arcgeosimr")
    unset(${TARGET}_LIBRARIES)
    # Build ArcGeoSim library list for dll copy
    get_target_property(${target}_LIBRARIES ${target} INTERFACE_LINK_LIBRARIES)
    foreach (lib ${${target}_LIBRARIES})
      string(REPLACE "ArcGeoSim::" "" lib_name ${lib})
	  message(STATUS "find_library ${lib}_LIBRARIES ${lib_name} HINTS ${ARCGEOSIM_FRAMEWORK_ROOT}/lib")
      find_library(${lib}_LIBRARIES ${lib_name} HINTS ${ARCGEOSIM_FRAMEWORK_ROOT}/lib)
      list(APPEND ${TARGET}_LIBRARIES ${${lib}_LIBRARIES})
    endforeach ()
    message(STATUS "ArcGeoSim libraries ${${TARGET}_LIBRARIES}")
  endif ()

  foreach(lib ${${TARGET}_LIBRARIES})
    if(${lib} STREQUAL optimized)
      continue()
    endif()
    if(${lib} STREQUAL debug)
      continue()
    endif()
    get_filename_component(dll ${lib} NAME_WE)
    get_filename_component(path ${lib} PATH)
    if("${path}" STREQUAL "") # lib locale ? lib systeme ?
      continue()
    endif()
    get_filename_component(extension ${lib} EXT) # Handle static lib
    file(GLOB dlls "${path}/*${dll}*.dll")
    if ("${dlls}" STREQUAL "" AND ${TARGET} STREQUAL "MKL") # patch for mkl, dll are in redist directory
      string(REPLACE "windows/mkl/lib/intel64" "windows/redist/intel64_win/mkl" path_dlls ${path})
      set(path ${path_dlls})
      file(GLOB dlls "${path}/*${dll}*.dll")
    elseif ("${dlls}" STREQUAL "" AND ${TARGET} STREQUAL "IFPSOLVER") # patch for ifpsolver static lib
      file(GLOB dlls "${path}/*${dll}*.a")
    endif ()

    get_filename_component(path ${path} PATH)
    if("${path}" STREQUAL "") # lib locale ? lib systeme ?
      continue()
    endif()
    file(GLOB lib_dlls "${path}/lib/*${dll}*.dll")
    file(GLOB bin_dlls "${path}/bin/*${dll}*.dll")
    if(NOT MSVC_TOOLSET_VERSION GREATER_EQUAL 143)
      file(GLOB bin_dlls2 "${path}/../bin/*${dll}*.dll")
    endif()  
    list(APPEND dlls  ${bin_dlls} ${bin_dlls2} ${lib_dlls})
    list(REMOVE_DUPLICATES dlls)
    foreach(dll ${dlls})
      copyOneDllFile(${dll})
    endforeach()
  endforeach()

  # Sometimes target or TARGET. Todo : refactor, at least create a function
  foreach(lib ${${target}_LIBRARIES})
    if(${lib} STREQUAL optimized)
      continue()
    endif()
    if(${lib} STREQUAL debug)
      continue()
    endif()
    get_filename_component(dll ${lib} NAME_WE)
    get_filename_component(path ${lib} PATH)
    if("${path}" STREQUAL "") # lib locale ? lib systeme ?
      continue()
    endif()
    file(GLOB dlls "${path}/*${dll}*.dll")
    get_filename_component(path ${path} PATH)
    if("${path}" STREQUAL "") # lib locale ? lib systeme ?
      continue()
    endif()
    file(GLOB lib_dlls "${path}/lib/*${dll}*.dll")
    file(GLOB bin_dlls "${path}/bin/*${dll}*.dll")
    if(NOT MSVC_TOOLSET_VERSION GREATER_EQUAL 143)
      file(GLOB bin_dlls2 "${path}/../bin/*${dll}*.dll")
    endif()
    list(APPEND dlls ${bin_dlls} ${bin_dlls2} ${lib_dlls})
    list(REMOVE_DUPLICATES dlls)
    foreach(dll ${dlls})
      copyOneDllFile(${dll})
    endforeach()
  endforeach()
  
endfunction()
