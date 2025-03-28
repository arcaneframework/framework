set(SOURCES
  ISerializer.h
  BasicSerializer.h
  BasicSerializer.cc
  SerializeGlobal.h
  SerializeGlobal.cc
  internal/BasicSerializerInternal.h
)

arccore_add_component_library(serialize
  FILES ${SOURCES}
)

target_link_libraries(arccore_serialize PUBLIC arccore_collections arccore_base)
