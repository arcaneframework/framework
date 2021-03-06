
# Macro pour ajouter récursivement à '_ALLLIBS' les bibliothèques
# 'lib' et ses dépendances en se basant sur la valeur de ${lib}_DEPENDS
set(_ALLLIBS)
set(_DONE_TARGETS)
macro(arcane_vtkutils_add_depend_lib_to_list lib)
  if (TARGET ${lib})
    list(FIND _DONE_TARGETS ${lib} _IDX)
    if (${_IDX} EQUAL -1)
      list(APPEND _DONE_TARGETS ${lib})
      #message(STATUS "ADD_DEPEND_LIB_TO_LIST LIB=${lib}")
      get_target_property(_LOC1 ${lib} LOCATION)
      #message(STATUS "  LOCATION=${_LOC1}")
      list(APPEND _ALLLIBS ${_LOC1})
      #message(STATUS " DEPENDS = ${${lib}_DEPENDS}")
      foreach(sublib ${${lib}_DEPENDS})
        arcane_vtkutils_add_depend_lib_to_list(${sublib})
      endforeach()
    endif()
  endif()
endmacro()
