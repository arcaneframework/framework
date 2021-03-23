set(EXAMPLE_NAME Poisson)

project(Poisson LANGUAGES C CXX)
include(../samples.common.cmake)

add_test(NAME poisson1 COMMAND Poisson Poisson.arc)
