set(EXAMPLE_NAME Geometry)

project(Geometry LANGUAGES C CXX)
include(../samples.common.cmake)

add_test(NAME geometry1 COMMAND Geometry Geometry.arc)
