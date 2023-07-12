macro(manageMetasActivation)
  
  if(EnableMetas)
    
    string(REPLACE "/" ";" ${EnableMetas} ${EnableMetas})

    foreach(meta ${EnableMetas})
      loadMeta(NAME ${meta})    
    endforeach()

  endif()
  
endmacro()
