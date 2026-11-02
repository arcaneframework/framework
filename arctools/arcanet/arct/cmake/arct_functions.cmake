if(NOT DEFINED ARCT_FUNCTIONS)
  set(ARCT_FUNCTIONS ON)

  include(${CMAKE_CURRENT_LIST_DIR}/arct_edit.cmake)
  include(${CMAKE_CURRENT_LIST_DIR}/arct_to_ctest.cmake)
endif ()