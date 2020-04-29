# Alien: generic linear algebra wrapper

This repository contains the "main" component of Alien.

## Description

TODO: fonctionalites principales d'Alien.

## Using Alien

### Spack

```shell script
spack install alien+hypre+superlu+hdf5
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

Alien also requires some of its framework components already installed and available for CMake (via CMAKE_PREFIX_PATH 
or <PACKAGE>_ROOT):
 - Arccon, the framework centralized build system;
 - Arccore, the base data structures, compiled with Arrays and MPI supports.

### Compiling

Alien relies on CMake.

```shell script
cmake <ALIEN_SRC_PATH>
cmake --build . 
```

#### Important note for debian/ubuntu

On debian and ubuntu systems, dynamic linker is lazy: if symbols are not explicitly referenced, the corresponding 
library is not linked, nor referenced.
This breaks our plugin frameworks, so, it is necessary to pass the following option to CMake.
```shell script
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed"
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
