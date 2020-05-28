#
# Find the BOOST includes and library
#
# This module uses
# BOOST_ROOT
#
# This module defines
# BOOST_FOUND
# BOOST_INCLUDE_DIRS
# BOOST_LIBRARIES
#
# Target boost

if(NOT BOOST_ROOT)
  set(BOOST_ROOT $ENV{BOOST_ROOT})
endif()


if(BOOST_ROOT)
  set(_BOOST_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_BOOST_SEARCH_OPTS)
endif()

# NB: sous linux, les chemins système passent malgré
# l'option Boost_NO_SYSTEM_PATHS...
if(NOT WIN32)
  if(BOOST_ROOT)
    set(BOOST_INCLUDEDIR ${BOOST_ROOT}/include)
    set(BOOST_LIBRARYDIR ${BOOST_ROOT}/lib)
    set(Boost_NO_SYSTEM_PATHS ON) 
  else()
    set(Boost_NO_SYSTEM_PATHS OFF)
  endif()
else()
  set(Boost_NO_SYSTEM_PATHS ON)
endif()

set(Boost_USE_STATIC_LIBS          OFF)
set(Boost_USE_MULTITHREADED        OFF) 
set(Boost_USE_STATIC_RUNTIME       OFF)
set(Boost_DETAILED_FAILURE_MSG     OFF)
set(Boost_PROGRAM_OPTIONS_DYN_LINK ON)
set(Boost_SYSTEM_DYN_LINK          ON)
set(Boost_THREAD_DYN_LINK          ON)
set(Boost_SERIALIZATION_DYN_LINK   ON)
set(Boost_CHRONO_DYN_LINK          ON)
set(Boost_REGEX_DYN_LINK           ON)
set(Boost_NO_BOOST_CMAKE           ON)

set(Boost_DEBUG OFF)

if(Boost_FIND_COMPONENTS)
find_package(Boost
        COMPONENTS ${Boost_FIND_COMPONENTS}
        QUIET)
else(Boost_FIND_COMPONENTS)
  find_package(Boost QUIET)
endif(Boost_FIND_COMPONENTS)

set(REQUIRED_COMPONENTS)
set(l_components)
foreach(COMPONENT ${Boost_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} COMPO_NAME)
  string(TOLOWER ${COMPONENT} compo_name)
  list(APPEND REQUIRED_COMPONENTS Boost_${COMPO_NAME})
  list(APPEND l_components ${compo_name})
endforeach(COMPONENT ${Boost_FIND_COMPONENTS})

logStatus("Boost_FOUND = ${Boost_FOUND}")
foreach(VAR ${REQUIRED_COMPONENTS})
  logStatus("${VAR}_FOUND = ${${VAR}_FOUND}")
endforeach()



# pour limiter le mode verbose
set(BOOST_FIND_QUIETLY ON)

find_package_handle_standard_args(BOOST
  DEFAULT_MSG 
  Boost_FOUND 
  Boost_INCLUDE_DIR)


logStatus("BOOST_FOUND = ${BOOST_FOUND}")

if (NOT BOOST_FOUND)
  return()
endif(NOT BOOST_FOUND)

set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIR})

set(BOOST_LIBRARIES)
foreach(VAR ${REQUIRED_COMPONENTS})
  list(APPEND BOOST_LIBRARIES ${${VAR}_LIBRARY})
endforeach()

if(NOT TARGET boost)
  logStatus("Boost is found, defining target boost")

#  if(NOT TARGET Boost::boost)

  add_library(boost INTERFACE IMPORTED)

  foreach(compo_name ${l_components})
    set(lib_compo "boost_${compo_name}")
    string(TOUPPER ${compo_name} COMPO_NAME)
    add_library(${lib_compo} UNKNOWN IMPORTED)

    set_target_properties(${lib_compo} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${BOOST_INCLUDE_DIRS}")

    set_property(TARGET ${lib_compo} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(${lib_compo} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
            IMPORTED_LOCATION_RELEASE "${Boost_${COMPO_NAME}_LIBRARY_RELEASE}")

    set_property(TARGET ${lib_compo} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)

    set_target_properties(${lib_compo} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
            IMPORTED_LOCATION_DEBUG "${Boost_${COMPO_NAME}_LIBRARY_DEBUG}")

    set_target_properties(${lib_compo} PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")

    set_property(TARGET ${lib_compo} APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS "BOOST_PROGRAM_OPTIONS_DYN_LINK")

    #add to boost target
    set_property(TARGET boost APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES ${lib_compo})
  endforeach(compo_name ${l_components})
#  else(NOT TARGET Boost::boost)
#    add_library(boost INTERFACE IMPORTED)
#    set_property(TARGET boost APPEND PROPERTY
#            INTERFACE_LINK_LIBRARIES Boost::boost)
#    #add_library(boost ALIAS Boost::boost)
#    foreach(compo_name ${l_components})
#      add_library(boost_${compo_name} INTERFACE IMPORTED)
#      set_property(TARGET boost_${compo_name} APPEND PROPERTY
#              INTERFACE_LINK_LIBRARIES Boost::${compo_name})
#      #add_library(boost_${compo_name} ALIAS Boost::${compo_name})
#    endforeach(compo_name ${l_components})
#  endif (NOT TARGET Boost::boost)
endif()
