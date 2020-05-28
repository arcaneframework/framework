#
# Find the GTest includes and library
#

if(NOT GTEST_ROOT)
  set(GTEST_ROOT $ENV{GTEST_ROOT})
endif()

if(GTEST_ROOT)
  set(_GTEST_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_GTEST_SEARCH_OPTS)
endif()

# pour limiter le mode verbose
set(GTest_FIND_QUIETLY ON)

set(_cmake_gtest false)
find_package(GTest)

if(NOT GTEST_FOUND)

  find_library(GTEST_LIBRARY_DEBUG
    NAMES gtest
    HINTS ${GTEST_ROOT} 
		PATH_SUFFIXES lib/Debug lib
    ${_GTEST_SEARCH_OPTS}
    )
  mark_as_advanced(GTEST_LIBRARY_DEBUG)

  find_library(GTEST_LIBRARY_RELEASE
    NAMES gtest
    HINTS ${GTEST_ROOT} 
		PATH_SUFFIXES lib/Release lib
    ${_GTEST_SEARCH_OPTS}
    )
  mark_as_advanced(GTEST_LIBRARY_RELEASE)

  find_library(GTEST_MAIN_LIBRARY_DEBUG
    NAMES gtest_main
    HINTS ${GTEST_ROOT} 
		PATH_SUFFIXES lib/Debug lib
    ${_GTEST_SEARCH_OPTS}
    )
  mark_as_advanced(GTEST_MAIN_LIBRARY_DEBUG)

  find_library(GTEST_MAIN_LIBRARY_RELEASE
    NAMES gtest_main
    HINTS ${GTEST_ROOT} 
		PATH_SUFFIXES lib/Release lib
    ${_GTEST_SEARCH_OPTS}
    )
  mark_as_advanced(GTEST_MAIN_LIBRARY_RELEASE)

  find_path(GTEST_INCLUDE_DIR gtest/gtest.h
    HINTS ${GTEST_ROOT} 
		PATH_SUFFIXES include
    ${_GTEST_SEARCH_OPTS}
    )
  mark_as_advanced(GTEST_INCLUDE_DIR)

  find_package_handle_standard_args(GTest
    DEFAULT_MSG
    GTEST_INCLUDE_DIR
    GTEST_LIBRARY_RELEASE
    GTEST_LIBRARY_DEBUG
    GTEST_MAIN_LIBRARY_DEBUG
    GTEST_MAIN_LIBRARY_RELEASE)

endif(NOT GTEST_FOUND)

if(GTEST_FOUND)
  
  set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
  
  include(CMakeFindDependencyMacro)
 
  # pour limiter le mode verbose
  set(Threads_FIND_QUIETLY ON)

  find_dependency(Threads QUIET)

  if(NOT TARGET GTest::GTest)
     
    add_library(GTest::GTest UNKNOWN IMPORTED)
	    
    set_target_properties(GTest::GTest PROPERTIES 
	    INTERFACE_LINK_LIBRARIES "Threads::Threads")
      
	  set_target_properties(GTest::GTest PROPERTIES 
	    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}")
      
	  set_property(TARGET GTest::GTest APPEND PROPERTY
      IMPORTED_CONFIGURATIONS RELEASE)
      
	  set_target_properties(GTest::GTest PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
      IMPORTED_LOCATION_RELEASE "${GTEST_LIBRARY_RELEASE}")
      
    set_property(TARGET GTest::GTest APPEND PROPERTY
      IMPORTED_CONFIGURATIONS DEBUG)
      
	  set_target_properties(GTest::GTest PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
      IMPORTED_LOCATION_DEBUG "${GTEST_LIBRARY_DEBUG}")
      
    # on enlève les tuples pour windows...
    set_target_properties(GTest::GTest PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "GTEST_USE_OWN_TR1_TUPLE=0")
    set_property(TARGET GTest::GTest APPEND PROPERTY
      INTERFACE_COMPILE_DEFINITIONS "GTEST_HAS_TR1_TUPLE=0")

  endif()

  if(NOT TARGET gtest)
    
    # librarie gtest
 
    add_library(gtest UNKNOWN IMPORTED)
	    
    set_target_properties(gtest PROPERTIES 
	    INTERFACE_LINK_LIBRARIES "GTest::GTest")
    
  endif()
 
endif()

