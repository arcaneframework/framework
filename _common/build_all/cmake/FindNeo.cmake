if (NOT Arctools_ROOT)
    # Pour compatibilité
    set(Arctools_ROOT "${ARCTOOLS_ROOT}")
endif()
if (NOT Arctools_ROOT)
    message(FATAL_ERROR "Variable 'Arctools_ROOT' is not set")
endif()

set(Neo_ROOT ${Arctools_ROOT}/neo)

if(NOT Neo_ALREADY_FOUND)
    add_subdirectory(${Neo_ROOT} neo)
    message(STATUS "Neo FOUND")
    set(Neo_ALREADY_FOUND YES PARENT_SCOPE)
endif()
set(Neo_FOUND YES)