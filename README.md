# Alien

All Alien world into one git repository.
This repository aims at easing development by compiling together Alien's core, its high level dependencies and its plugins.

## How to use ?

Simply clone this repository.

```shell script
git clone https://github.com/arcaneframework/alien.git
cd alien
```

After, you can run CMake (at least version 3.15).
```shell script
mkdir build && cd build
cmake ..
```

Useful CMake options:
- ALIEN_FRAMEWORK_EXTERNAL, compile each subproject separately, OFF by default
- ALIEN_PLUGIN_HYPRE, whether Hypre plugin is compiled, OFF by default
- ALIEN_PLUGIN_SUPERLU, whether SuperLUDist plugin is compiled, OFF by default
- ALIEN_USE_HDF5, whether HDF5 support is enabled, OFF by default.

We can mention also other generic CMake options :
- CMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed", useful on debian based linux distribution (like ubuntu), 
as without linker drops libraries that are not explicitly referenced, breaking our plugin interface.
- CMAKE_VERBOSE_MAKEFILE=ON

## Requirements

Alien requires a recent build environment:
 - recent CMake (>= 3.15)
 - C++ compiler that supports C++-14 (gcc, llvm/clang, intel)
 - MPI
 - boost, at least with timer and program options components enabled
 - glib2
 - BLAS
 - Google Tests, for unit tests.
 
 On Ubuntu-20.04, installing this package is sufficient:
 ```shell script
apt-get install build-essential cmake gcc g++ gdb \
        libhypre-dev libsuperlu-dist-dev \
        libboost-dev libboost-program-options-dev libgtest-dev libglib2.0-dev
```

For GoogleTest, one must finish installation by running:
```shell script
cd $(mktemp -d) && cmake /usr/src/googletest && cmake --build . --target install
```

## How it works ?

This repository contains several other repositories, needed for Alien. 
It contains the following subdirectories:
 - framework: from Arcane, these subdirectories are synchronized by `git subtree`
   + arccon,  which contains our build system, based on CMake;
   + arccore, which contains base features on which Alien is built, mainly Array and ParallelManager;
 - src, the main repository for linear algebra,
 - plugins, with different plugins for Alien, to call Hypre or SuperLU external libraries.
