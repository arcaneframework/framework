.. _developer_doxygen:

Alien's Coding Style
====================

Naming
------

 - CamelCase for c++ objects
 - underscore_separator_and_no_capital for filenames

C++
---

Alien is (at the moment) a C++ library. We follow directives of `Arcane`, mainly:
 - C++14 code
 - Avoid passing parameters by reference (specially it is forbidden for all *Array* classes).

File Organization
-----------------

Alien consists in different modules:
  - `core`
  - `movesemantic`
  - `refsemantic`

Modules are located in the `src` directory and each module is organized following this directory structure:
  - `alien` directory, that mimics header hierarchy and contains both header files and source files
  - `tests` directory, that contains unit tests for this module.

.. code-block:: bash

   .
   ├── CMake
   ├── docker             # Contains docker images for CI and development
   ├── docs               # Documentation and examples
   │   ├── sphinx   # This Sphinx documentation
   │   └── tutorial # Alien's tutorial
   ├── framework          # Arccon and Arccore for embedded build
   │   ├── arccon   # Arccon
   │   ├── arccore  # Arccore
   │   └── CMake    # CMake glue for find_package to work with embedded
   ├── plugins            # Alien's backend
   │   ├── hypre
   │   └── petsc
   └── src                # Alien's sources
      ├── core            # Alien core module
      │   ├── alien # Sources for `core` module
      │   └── tests # Unit tests for `core` module
      ├── movesemantic    # `move` API
      │   ├── alien # Sources for `move` API
      │   └── tests # Unit tests for `move`
      ├── refsemantic     # `ref` API
      │   ├── alien # Sources for `ref` API
      │   └── tests # Unit tests for `ref`
      └── test_framework  # Unit test framework


