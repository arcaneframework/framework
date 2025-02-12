set(SOURCES
  Array.h
  Array.cc
  Array2.h
  Array2.cc
  ArrayTraits.h
  ArrayDebugInfo.h
  CollectionsGlobal.h
  CollectionsGlobal.cc
  IMemoryAllocator.h
  MemoryAllocationArgs.h
  MemoryAllocationOptions.h
  MemoryAllocationOptions.cc
  MemoryAllocator.cc
  StringVector.h
  StringVector.cc
)

arccore_add_component_library(collections
  FILES ${SOURCES}
)

target_link_libraries(arccore_collections PUBLIC arccore_base)
