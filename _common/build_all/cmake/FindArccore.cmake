if (NOT Arccore_ROOT)
  # Pour compatibilité
  set(Arccore_ROOT "${ARCCORE_ROOT}")
endif()
if (NOT Arccore_ROOT)
   message(FATAL_ERROR "Variable 'Arccore_ROOT' is not set")
endif()
if (NOT FRAMEWORK_NO_EXPORT_PACKAGES)
  if (NOT ARCCORE_EXPORT_TARGET)
    # Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des
    # packages définies dans 'Arccon'.
    set(ARCCON_EXPORT_TARGET ArcaneTargets)
  endif()
endif()

add_subdirectory(${Arccore_ROOT} arccore)
set(Arccore_FOUND YES)
set(Arccore_FOUND YES PARENT_SCOPE)
