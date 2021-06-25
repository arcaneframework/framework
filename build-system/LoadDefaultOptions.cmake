# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Verbose mode
createOption(COMMANDLINE Verbose
             NAME        VERBOSE
             MESSAGE     "Verbosity" 
             DEFAULT     OFF)

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  set(CMAKE_REQUIRED_QUIET OFF)
else()
  set(CMAKE_VERBOSE_MAKEFILE OFF)
  set(CMAKE_REQUIRED_QUIET ON)
endif()
 
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# mode C++11
createOption(COMMANDLINE Cxx11
             NAME        USE_CXX11 
             MESSAGE     "C++11 standard" 
             DEFAULT     ON)

if(USE_CXX11) 

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF) # c++11 ou lieu de gnu++11

  # Liste des fonctionnalités du C++11 utilisées par défaut.
  # Cette liste est un sous-ensemble de la variable CMAKE_CXX_KNOWN_FEATURES
  # GG NOTE: Actuellement (version 2.2), on utilise les fonctionnalités suivantes
  # du C++11: cxx_inline_namespaces, cxx_deleted_functions.
  # Cependant, avec cmake 3.3.2, ces features sont considérées comme
  # non supportées par Visual Studio 2013 alors que ce n'est pas le cas
  # (avec VS2013 Update 4, cela fonctionne). On les enlève donc provisoirement
  # de la liste. A noter qu'avec VS2015, ces fonctionnalités sont correctement
  # reconnuees.
  set(CXX11_FEATURES cxx_lambdas cxx_auto_type
                     cxx_nullptr cxx_rvalue_references)
 
  # on définit le meta
  loadMeta(NAME c++11)
endif()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if(NOT WIN32)

  # Wall optimization
  createOption(COMMANDLINE Wall
               NAME        USE_WALL
               MESSAGE     "Warning compilation flag (-Wall)" 
               DEFAULT     ON)

  if(USE_WALL)
    appendCompileOption(FLAG Wall)
  endif()

endif()

# SSE optimization
createOption(COMMANDLINE SSE
             NAME        USE_SSE
             MESSAGE     "SSE compilation flags (-msse2 -msse)" 
             DEFAULT     ON)

if(USE_SSE AND NOT WIN32)
  if(${CMAKE_C_COMPILER_ID} STREQUAL GNU)
    appendCompileOption(FLAG mfpmath=sse msse msse2 CONFIGURATION RELEASE)
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL Clang)
    appendCompileOption(FLAG msse msse2 CONFIGURATION RELEASE)
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL Intel)
    appendCompileOption(FLAG msse3 w1 vec-report0 CONFIGURATION RELEASE)
  endif()
endif()
if(USE_SSE AND WIN32)
  appendCompileOption(FLAG Oi CONFIGURATION RELEASE)
endif()

if(NOT WIN32)
  # AVX optimization
  createOption(COMMANDLINE AVX
               NAME        USE_AVX
               MESSAGE     "AVX compilation flag (-mavx)" 
               DEFAULT     OFF)

  if(USE_AVX AND NOT WIN32)
    if(${CMAKE_C_COMPILER_ID} STREQUAL GNU)
      appendCompileOption(FLAG mavx)
    endif()
  endif()
endif()
  
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# debug
createOption(COMMANDLINE Debug
             NAME        DEBUG 
             MESSAGE     "Debug mode" 
             DEFAULT     ON)

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type mode (Debug or Release)" FORCE)
else()
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type mode (Debug or Release)" FORCE)
endif()

# on ne veut que la configuration courante
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE TYPE INTERNAL FORCE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
 
if(WIN32)

  # force multiple WIN32
  createOption(COMMANDLINE ForceMultiple
               NAME       FORCE_MULTIPLE 
               MESSAGE    "Remove link conflicts (multiple symbols) : /force:multiple for windows VS" 
               DEFAULT    ON)
  
  if(FORCE_MULTIPLE)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG   "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}   /FORCE:MULTIPLE")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG      "${CMAKE_EXE_LINKER_FLAGS_DEBUG}      /FORCE:MULTIPLE")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /FORCE:MULTIPLE")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE}    /FORCE:MULTIPLE")
  endif()
  
endif()
   
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
 
if(WIN32)

  # remove default lib WIN32
  createOption(COMMANDLINE RemoveCMT
               NAME        REMOVE_CMT 
               MESSAGE     "Remove runtime (mix /MT,/MD dlls) : /nodefaultlib:libcmt for windows VS" 
               DEFAULT     ON)
  
  if(REMOVE_CMT)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG   "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}   /NODEFAULTLIB:LIBCMT")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG      "${CMAKE_EXE_LINKER_FLAGS_DEBUG}      /NODEFAULTLIB:LIBCMT")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE}    /NODEFAULTLIB:LIBCMT")
  endif()
  
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(WIN32)

  # allows parallel compilation
  createOption(COMMANDLINE ParallelCompilation
               NAME        PARALLEL_COMPILATION 
               MESSAGE     "Parallel compilation : /MP for windows VS" 
               DEFAULT     ON)
  
  if(PARALLEL_COMPILATION)
    set(CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS}   /MP")
    set(CMAKE_C_FLAGS      "${CMAKE_C_FLAGS}     /MP")
  endif()
  
endif()
 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# shared
createOption(COMMANDLINE Shared
             NAME        SHARED 
             MESSAGE     "Shared build mode" 
             DEFAULT     ON)

# shared libraries
if(SHARED)
  set(BUILD_SHARED_LIBS True CACHE Bool "Shared build mode (True or False)" FORCE)
else()
  set(BUILD_SHARED_LIBS False CACHE Bool "Shared build mode (True or False)" FORCE)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Package file
if(BUILDSYSTEM_DEFAULT_PACKAGE_FILE)
  # on a un défaut
  createStringOption(COMMANDLINE PackageFile
                     NAME        PACKAGE_FILE
                     MESSAGE     "File defining roots of packages" 
                     DEFAULT     ${BUILDSYSTEM_DEFAULT_PACKAGE_FILE})
else()
  # ou non 
  createStringOption(COMMANDLINE PackageFile
                     NAME        PACKAGE_FILE
                     MESSAGE     "File defining roots of packages")
endif()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
