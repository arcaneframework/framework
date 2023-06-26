
logStatus(" ** Generate alien configuration files :")

logStatus("  * AlienLegacyConfig.h")

get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)

foreach(target ${TARGETS})
  if(${target}_IS_LOADED)
    string(TOUPPER ${target} name)
    set(ALIEN_USE_${name} ON)
  endif()
endforeach()

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/AlienLegacyConfig.h.in 
  ${PROJECT_BINARY_DIR}/alien/AlienLegacyConfig.h
  )

install(
  FILES ${PROJECT_BINARY_DIR}/alien/AlienLegacyConfig.h
  DESTINATION include/alien
  )