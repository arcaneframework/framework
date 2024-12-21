cmake_minimum_required(VERSION 3.11)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()
logStatus(" **  Project name      : ${BoldRed}${PROJECT_NAME}${ColourReset}")
logStatus(" **          version   : ${BoldRed}${PROJECT_VERSION}${ColourReset}")
message_separator()
logStatus(" **  System name       : ${CMAKE_SYSTEM_NAME}")
logStatus(" **         version    : ${CMAKE_SYSTEM_VERSION}")
logStatus(" **         processor  : ${CMAKE_SYSTEM_PROCESSOR}")
if(EXISTS "/etc/redhat-release")
  file(READ "/etc/redhat-release" REDHAT_RELEASE)
  string(REPLACE "\n" "" REDHAT_RELEASE ${REDHAT_RELEASE})
  logStatus(" **         vendor     : ${REDHAT_RELEASE}")
endif()
message_separator()
site_name(BUILD_SITE_NAME)
logStatus(" ** Build site name    : ${BUILD_SITE_NAME}")
message_separator()
logStatus(" **  Generator         : ${CMAKE_GENERATOR}")
message_separator()
logStatus(" **  Build System path : ${BUILD_SYSTEM_PATH}")
logStatus(" **       Install path : ${CMAKE_INSTALL_PREFIX}")
logStatus(" **     Dlls copy path : ${BUILDSYSTEM_DLL_COPY_DIRECTORY}")
if(BUILDSYSTEM_NO_CONFIGURATION_OUTPUT_DIRECTORY)
  logStatus(" ** No configuration in output directories lib/bin")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
