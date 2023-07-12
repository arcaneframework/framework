message_separator()

logStatus("Load ArcGeoSim project configuration (defines, includes, etc.)")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(USE_CXX11)
  add_definitions(-DARCGEOSIM_USE_CXX11)
else()
  logFatalError("ArcGeoSim needs C++11 standard")
endif()
add_definitions(-DARCGEOSIM_USE_EXPORT)
add_definitions(-DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX)

if(WIN32)
  # tous les symboles exportes sous windows
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# includes pour simplifier
include_directories(${PROJECT_BINARY_DIR})
include_directories(${ARCGEOSIM_FRAMEWORK_ROOT}/ArcGeoPhy/src)
include_directories(${ARCGEOSIM_FRAMEWORK_ROOT}/ArcGeoSim/src)
include_directories(${ARCGEOSIM_FRAMEWORK_ROOT}/SharedUtils/src)
include_directories(${ARCGEOSIM_FRAMEWORK_ROOT}/ArximCpp/src)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
