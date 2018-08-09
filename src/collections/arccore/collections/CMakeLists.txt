find_package(Glib REQUIRED)

set(SOURCES
  Array.h
  Array.cc
  Array2.h
  Array2.cc
  CollectionsGlobal.h
  IMemoryAllocator.h
  MemoryAllocator.cc
)

arccore_add_library(arccore_collections
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/collections
  RELATIVE_PATH arccore/collections
  FILES ${SOURCES}
)

target_compile_definitions(arccore_collections PRIVATE ARCCORE_COMPONENT_arccore_collections)
target_include_directories(arccore_collections PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/collections> $<INSTALL_INTERFACE:include>)
install(TARGETS arccore_collections EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)

target_link_libraries(arccore_collections PUBLIC arccore_base)
