# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
  # -g pour CC et CXX
  appendCompileOption(FLAG g) 
  # -ansi pour CC
  appendCompileOption(FLAG ansi LANGUAGE CC) 
  # -march=native pour CC et CXX (GNU)
  #if(${CMAKE_C_COMPILER_ID} STREQUAL GNU)
  #  appendCompileOption(FLAG march=native)
  #endif()
  # -fno-check-new pour CXX (GNU et INTEL)
  if(${CMAKE_C_COMPILER_ID} STREQUAL GNU OR ${CMAKE_C_COMPILER_ID} STREQUAL Intel)
    appendCompileOption(FLAG fno-check-new LANGUAGE CXX)
  endif()
  # -fno-builtin pour CXX
  appendCompileOption(FLAG fno-builtin LANGUAGE CXX)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_definitions(-Darch_${CMAKE_SYSTEM_NAME})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

check_cxx_compiler_flag(${CMAKE_CXX_FLAGS_DEBUG}   compiler_supports_Debug_flags)
check_cxx_compiler_flag(${CMAKE_CXX_FLAGS_RELEASE} compiler_supports_Release_flags)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(NOT WIN32)
  logStatus(" ** Base CXX flags : ${CMAKE_CXX_FLAGS}")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    logStatus(" ** Debug mode enabled : ${CMAKE_CXX_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    logStatus(" ** Release mode enabled : ${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
else()
  logStatus(" **   Debug flags : ${CMAKE_CXX_FLAGS_DEBUG}")
  logStatus(" ** Release flags : ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(${BUILD_SHARED_LIBS})
  logStatus(" ** Shared build mode enabled")
else()
  logStatus(" ** Static build mode enabled")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
