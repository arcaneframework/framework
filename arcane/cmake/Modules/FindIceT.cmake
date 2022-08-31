arccon_return_if_package_found(IceT)

find_library(IceT_CORE_LIBRARY IceTCore)
find_library(IceT_MPI_LIBRARY IceTMPI)
find_library(IceT_GL_LIBRARY IceTGL)

message(STATUS "IceT_CORE_LIBRARY = ${IceT_CORE_LIBRARY}")
message(STATUS "IceT_MPI_LIBRARY = ${IceT_MPI_LIBRARY}")
message(STATUS "IceT_GL_LIBRARY = ${IceT_GL_LIBRARY}")

find_path(IceT_INCLUDE_DIR IceT.h)

message(STATUS "IceT_INCLUDE_DIR = ${IceT_INCLUDE_DIR}")
 
set(IceT_FOUND FALSE)
if(IceT_INCLUDE_DIR AND IceT_CORE_LIBRARY AND IceT_GL_LIBRARY AND IceT_MPI_LIBRARY)
  message(STATUS "IceT Found")
  set(IceT_FOUND TRUE)
  set(IceT_LIBRARIES ${IceT_MPI_LIBRARY} ${IceT_GL_LIBRARY} ${IceT_CORE_LIBRARY})
  set(IceT_INCLUDE_DIRS ${IceT_INCLUDE_DIR})
endif()

arccon_register_package_library(IceT IceT)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
