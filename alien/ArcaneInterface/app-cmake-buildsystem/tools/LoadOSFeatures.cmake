if(WIN32)

  # Check if VS available
  # On pourrait regarder en fonction du générateur aussi
  
  if(NOT DEFINED ENV{VSTUDIO_SHORT_NAME} OR NOT DEFINED ENV{VSTUDIO_NAME})
    logStatus("VSTUDIO_SHORT_NAME = $ENV{VSTUDIO_SHORT_NAME}")
    logStatus("VSTUDIO_NAME = $ENV{VSTUDIO_NAME}")
    logFatalError("Environment variables 'VSTUDIO_SHORT_NAME' and 'VSTUDIO_NAME' must be defined")
  endif()

  message_separator()
 
  logStatus("Visual studio information :")
  logStatus(" ** short name : $ENV{VSTUDIO_SHORT_NAME}")
  logStatus(" **       name : $ENV{VSTUDIO_NAME}")

else()

  if(NOT REDHAT_RELEASE)
    logFatalError("** This environment is not RedHat based **")
  endif()

  if(${REDHAT_RELEASE} MATCHES "release 5.")
    set(RHEL_TAG RHEL5)
    loadMeta(NAME rhel5)
  elseif(${REDHAT_RELEASE} MATCHES "release 6.")
    set(RHEL_TAG RHEL6)
    loadMeta(NAME rhel6)
  elseif(${REDHAT_RELEASE} MATCHES "release 7.")
    set(RHEL_TAG RHEL7)
    loadMeta(NAME rhel7)
  elseif(${REDHAT_RELEASE} MATCHES "release 8.")
    set(RHEL_TAG RHEL8)
    loadMeta(NAME rhel8)
  else()
    logFatalError("Cannot identify current RedHat release")
  endif()

endif()
