function(__generate_package_info target)
  set(buffer "")
  set(_Available "false")
  string(TOUPPER ${target} TARGET)
  if(${package}_IS_LOADED)
    set(_Available "true")
  endif()
  set(buffer "${buffer} <package name='${target}' available='${_Available}'>\n")
  if (_Available)
    foreach(lib ${${TARGET}_LIBRARIES})
	  if(NOT(${lib} STREQUAL "optimized" OR ${lib} STREQUAL "debug"))
        set(buffer "${buffer}   <lib-name>${lib}</lib-name>\n")
	  endif()
    endforeach(lib)
    foreach(inc ${${TARGET}_INCLUDE_DIRS})
      set(buffer "${buffer}   <include>${inc}</include>\n")
    endforeach(inc)
    foreach(exec ${${TARGET}_EXEC_PATH})
      set(buffer "${buffer}   <bin-path>${exec}</bin-path>\n")
    endforeach(exec)
    foreach(flags ${${TARGET}_FLAGS})
	  set(buffer "${buffer}   <flags>${flags}</flags>\n")
    endforeach(flags)
  endif()
  set(buffer "${buffer}  </package>\n\n")
  string(REPLACE "&" "&amp\;" buffer ${buffer})
  set(PKG_STR "${PKG_STR} ${buffer}" PARENT_SCOPE)
endfunction()

function(generatePackageXmlFile)

  set(PKG_STR "${PKG_STR}<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>\n\n")
  set(PKG_STR "${PKG_STR}<!-- Generated file DO NOT EDIT -->\n\n")
  set(PKG_STR "${PKG_STR}<packages>\n\n")
  
  get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)

  foreach(package ${TARGETS})
    __generate_package_info(${package})
  endforeach()

  set(PKG_STR "${PKG_STR}</packages>\n\n")

  file(WRITE ${PROJECT_BINARY_DIR}/pkglist.xml ${PKG_STR})

  install(FILES ${PROJECT_BINARY_DIR}/pkglist.xml
          DESTINATION lib
          )

endfunction()
