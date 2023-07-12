# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if(${PACKAGE_FILE})
  
  if(IS_ABSOLUTE ${PACKAGE_FILE_VALUE})
    set(file ${PACKAGE_FILE_VALUE})
  else()
    set(file ${PROJECT_BINARY_DIR}/${PACKAGE_FILE_VALUE})
  endif()

  if(NOT EXISTS ${file})
    logFatalError("Package file ${file} doesn't exist")
  endif()
  
  include(${file})

endif()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
