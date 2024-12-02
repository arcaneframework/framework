# ----------------------------------------------------------------------------
# Function to add one or several target or packages to a target
#
# Usage:
#
#  alien_link_target_or_package(target (PRIVATE|PUBLIC) <pkg1> <pkg2> ...)
#
# For example:
#
#  alien_link_target_or_package(alien_external_package PRIVATE HDF5 TBB)
#
# For each <pkg>, we try to determine the associated targets in the following order:
#
# 1. If <pkg> is a already a target, use it
# 2. If the variable ARCCON_PACKAGE_<pkg>_TARGETS is defined, its value is a list of target to test
# 2. If the variable ARCCON_TARGET_<pkg> is defined, its value is the name of the target
# 3. If arccon::<pkg> is a target use it.
#
# If the associated target for a package is found, we add it
# to $target with the command target_link_libraries. If it is not found,
# nothing is done.
#
function(alien_link_target_or_package target visibility)
  foreach(package ${ARGN})
    message(STATUS "Check raw '${package}'")
    if (TARGET ${package})
      set(_PKG ${package})
    elseif (DEFINED ARCCON_PACKAGE_${package}_TARGETS)
      # One or several target associated to this package
      message(STATUS "Checking adding several targets '${ARCCON_PACKAGE_${package}_TARGETS}'")
      alien_link_target_or_arccon_package(${target} ${visibility} ${ARCCON_PACKAGE_${package}_TARGETS})
      continue()
    elseif (DEFINED ARCCON_TARGET_${package})
      set(_PKG ${ARCCON_TARGET_${package}})
    else()
      set(_PKG arccon::${package})
    endif()
    message(STATUS "Check add package or target '${_PKG}'")
    if (TARGET ${_PKG})
      message(STATUS "Add target '${_PKG}'")
      target_link_libraries(${target} ${visibility} ${_PKG})
    endif()
  endforeach()
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
