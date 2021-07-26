if(Arccore_FOUND)
  return()
endif()

if (NOT ARCCORE_EXPORT_TARGET)
  # Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des
  # définies dans 'Arccore'.
  set(ARCCORE_EXPORT_TARGET ${FRAMEWORK_EXPORT_NAME})
endif()

# add directory only once !
if (NOT TARGET arccore_full)
  add_subdirectory(${ARCCORE_SRC_DIR} arccore)
endif(NOT TARGET arccore_full)
