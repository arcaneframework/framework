function(linkLibraries target)

  # commit ?
  get_target_property(committed ${target} BUILDSYSTEM_COMMITTED)

  if(${committed})
    logFatalError("target ${target} is already committed, can't link libraries")
  endif()

  # ajouts des libraries
  set_property(TARGET ${target} APPEND PROPERTY BUILDSYSTEM_LIBRARIES ${ARGN})
endfunction()

function(linkDynLibraries target)

  # commit ?
  get_target_property(committed ${target} BUILDSYSTEM_COMMITTED)

  if(${committed})
    logFatalError("target ${target} is already committed, can't link libraries")
  endif()

  # ajouts des libraries
  set_property(TARGET ${target} APPEND PROPERTY BUILDSYSTEM_LIBRARIES ${ARGN})
  set_property(TARGET ${target} APPEND PROPERTY NEED_DLOPEN ${ARGN})
endfunction()
