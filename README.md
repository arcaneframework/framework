# Alien: generic linear algebra wrapper

This repository contains the "main" component of Alien.

## Description

TODO: fonctionalites principales d'Alien.

## Using Alien

### Spack

```shell script
spack install alien
```

### AlienIntegratedFramework

See https://gitlab.com/AlienAlgebra/alien_integrated_framework/README.md

## Build Alien

Before using this repository, please check if previously described alternative methods do not better fit your need. 

### Requirements

Alien has the following generic direct dependencies:
 - a recent CMake (>3.15)
 - a C++ compiler that supports C++14
 - MPI
 - BLAS
 - Boost  
 - Arccon, the framework centralized build system
 - Arccore, the base data structures, compiled with Arrays and MPI supports.

### Optional dependencies

Optional features of Alien depends on:
 - libxml2, enabled with `ALIEN_USE_LIBXML2:BOOL=ON`
 - hdf5, enabled with `ALIEN_USE_HDF5:BOOL=ON`
 - GTest, for unit tests
 - doxygen, sphinx and breathe, for documentation generation.

### Compiling

Alien relies on CMake, the workflow for default options is the following:

```shell script
cmake <ALIEN_SRC_PATH>
cmake --build . 
```

#### Compile options

Alien build options to pass to CMake with `-D`:
 - ALIEN_COMPONENT_RefSemantic, enables build of the `ref` API
 - ALIEN_COMPONENT_MoveSemantic, enables build of the `move` API
 - ALIEN_UNIT_TESTS, enables unit tests (requires GTest)
 - ALIEN_GENERATE_DOCUMENTATION, builds documentation (requirements below)
 - ALIEN_USE_HDF5, use optional dependency hdf5
 - ALIEN_USE_LIBXML2, use optional dependency libxml2

In order to generate documentation, the following dependencies are mandatory:
 - doxygen
 - sphinx
 - sphinx-rtd-theme-common
 - breathe

On debian/ubuntu, this translates to:
```bash
sudo apt install doxygen python3-breathe python3-sphinx-rtd-theme
```

Breathe and sphinx can also be installed with `pip`:
```bash
pip install breathe
```

#### Important note for debian/ubuntu

On debian and ubuntu systems, dynamic linker is lazy: if symbols are not explicitly referenced, the corresponding 
library is not linked, nor referenced.
This breaks our plugin frameworks, so, it is necessary to pass the following option to CMake.
```shell script
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed"
```

#### Example

To build Alien and its documentation on ubuntu:
```bash
# Configure build
cmake -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed" -DALIEN_UNIT_TESTS:BOOL=ON -DALIEN_GENERATE_DOCUMENTATION:BOOL=ON <ALIEN_SRC_PATH>
# Build alien
cmake --build .
# Run tests
ctest
# Build documentation
cmake --build . --target alien_doc
```

## Organization

This repository contains 3 main modules for Alien.

### Main Modules

They are located in the modules subdirectory.

#### Core

The *core* directory contains main Alien library: alien_core.
It provides frontend and backend interfaces as well as multi-representations.
Any project using Alien should be linked to this library.

The Core module contains also basic kernels (i.e. linear algebra implementations of Alien), such as:
 - SimpleCSR, 
 - DoK (Dictionary of Keys),
 - Composite, for hierarchical linear algebra objects.
 - Redistribution for dynamically distributed MPI objects.

#### APIs

Alien comes with 2 standard APIs, for use in client codes:
 - the *move* semantic, where data ownership is transferred between objects, thanks to C++ std::move;
 - the *ref* semantic, where references are shared.
 
Any of these interfaces can be used with any backend.

#### Backends

Backends are available through separate repositories. They also depend on alien_core.
