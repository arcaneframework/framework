set(SOURCES
  MessagePassingGlobal.h
  MessagePassingGlobal.cc
  Request.h
  Request.cc
  IStat.h
  IDispatchers.h
  ITypeDispatcher.h
  IMessagePassingMng.h
  Stat.h
  Stat.cc
)

arccore_add_library(arccore_messagepassing
  INPUT_PATH ${Arccore_SOURCE_DIR}/src/messagepassing
  RELATIVE_PATH arccore/messagepassing
  FILES ${SOURCES}
)

target_compile_definitions(arccore_messagepassing PRIVATE ARCCORE_COMPONENT_arccore_messagepassing)
target_include_directories(arccore_messagepassing PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/messagepassing> $<INSTALL_INTERFACE:include>)
install(TARGETS arccore_messagepassing EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)

target_link_libraries(arccore_messagepassing PUBLIC arccore_collections arccore_base)
