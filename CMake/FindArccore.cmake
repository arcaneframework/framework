if (NOT ARCCORE_ROOT)
   message(FATAL_ERROR "Variable 'ARCCON_ROOT' is not set")
endif()
if (NOT ARCCORE_EXPORT_TARGET)
  # Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des
  # définies dans 'Arccore'.
  set(ARCCORE_EXPORT_TARGET ${FRAMEWORK_EXPORT_NAME})
endif()

# add directory only once !
if (NOT TARGET arccore_full)
  add_subdirectory(${ARCCORE_ROOT} arccore)
endif(NOT TARGET arccore_full)
