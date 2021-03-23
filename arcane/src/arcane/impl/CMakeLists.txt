add_include_directories_if_found(FLEXLM)

include(srcs.cmake)

set(PKGS FLEXLM)
if(FLEXLM_FOUND)
  set(ARCANE_SOURCES ${ARCANE_SOURCES} FlexLMTools.cc)
endif()

arcane_add_library(arcane_impl
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/impl
  AXL_FILES ${AXL_FILES}
  FILES ${ARCANE_SOURCES}
  )

arcane_add_arccon_packages(arcane_impl PRIVATE ${PKGS})

target_compile_definitions(arcane_impl PRIVATE ARCANE_COMPONENT_arcane_impl)

target_link_libraries(arcane_impl PUBLIC arcane_core)

# FlexLM
target_link_directories(arcane_utils PUBLIC ${FLEXLM_LIBRARY_DIRS})
target_link_libraries(arcane_utils PUBLIC ${FLEXLM_LIBRARIES})

arcane_register_library(arcane_impl)
