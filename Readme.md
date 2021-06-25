# This project is a build system for numerical software

It aims at running on Unix and Windows.
It requires CMake (>3.0).

Usage
-----

First, you need to include 'Arccon' root file. If ARCCON_ROOT contains
the path to 'Arccon', you can add the following line to your
CMakeLists.txt:

~~~~~~~~~
include(${ARCCON_ROOT}/Arccon.cmake)
~~~~~~~~~

The following packages are then availables via the command
'find_package':

- 'Mono': the open source .NET runtime 'mono'.
