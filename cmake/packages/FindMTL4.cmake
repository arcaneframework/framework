arccon_return_if_package_found(MTL4)

find_path(MTL4_INCLUDE_DIRS boost/numeric/mtl/mtl.hpp)

message(STATUS "MTL4_INCLUDE_DIRS=" ${MTL4_INCLUDE_DIRS})

unset(MTL4_FOUND)
if (MTL4_INCLUDE_DIRS)
    set(MTL4_FOUND TRUE)
    arccon_register_package_library(MTL4 MTL4)
endif ()
