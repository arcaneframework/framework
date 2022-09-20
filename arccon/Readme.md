# This project is a build system for numerical software

It aims at running on Unix and Windows.
It requires CMake 3.18+.

## Usage

~~~~~~~~~{cmake}
find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})
~~~~~~~~~
