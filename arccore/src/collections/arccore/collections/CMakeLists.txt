set(SOURCES
  Array.h
  Array.cc
  Array2.h
  Array2.cc
  CollectionsGlobal.h
  IMemoryAllocator.h
  MemoryAllocator.cc
)

arccore_add_component_library(collections
  FILES ${SOURCES}
)

target_link_libraries(arccore_collections PUBLIC arccore_base)
