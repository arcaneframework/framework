
logStatus(" ** Generate alien configuration files :")

logStatus("  * ALIENConfig.h")

get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)

foreach(target ${TARGETS})
  if(${target}_IS_LOADED)
    string(TOUPPER ${target} name)
    set(ALIEN_USE_${name} ON)
  endif()
endforeach()

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ALIENConfig.h.in 
  ${PROJECT_BINARY_DIR}/ALIEN/ALIENConfig.h
  )

install(
  FILES ${PROJECT_BINARY_DIR}/ALIEN/ALIENConfig.h
  DESTINATION include/ALIEN
  )