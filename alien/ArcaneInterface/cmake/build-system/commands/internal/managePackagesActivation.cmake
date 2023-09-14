macro(managePackagesActivation)
 
  if(DisablePackages)
    
    string(REPLACE "/" ";" DisablePackages "${DisablePackages}")
    
  endif()
  
  if(EnablePackages)
    
    string(REPLACE "/" ";" EnablePackages "${EnablePackages}")

    if (DisablePackages)
      # on retire les package
      foreach(package ${EnablePackages})
        list(REMOVE_ITEM DisablePackages ${package})
      endforeach()
      
      # on retire du cache
      unset(EnablePackages CACHE)
      # on met a jour le cache
      set(DisablePackages ${DisablePackages} CACHE STRING "Packages disabled" FORCE)
    endif()
    #
    foreach(package ${EnablePackages})
      EnablePackage(NAME ${package})
    endforeach()
  endif()
  
  foreach(package ${DisablePackages})
    disablePackage(
      NAME ${package}
      WHY "disabled by user (option DisablePackages)"
      )
  endforeach()

endmacro()
