# [Alien][alien]

All Alien world into one git repository. This repository aims at easing
development by compiling together Alien's core,
its high level dependencies and its plugins.

## How to use ?

Simply clone this repository.

```shell script
git clone https://github.com/arcaneframework/alien.git
cd alien
```

After, you can run CMake (at least version 3.18).

```shell script
mkdir build && cd build
cmake ..
```

Useful CMake options:

- ALIEN_FRAMEWORK_EXTERNAL, compile each subproject separately, OFF by default
- ALIEN_PLUGIN_GINKGO, whether [Ginkgo][ginkgo] plugin is compiled, OFF by
  default
- ALIEN_PLUGIN_HYPRE, whether [hypre][] plugin is compiled, OFF by default
- ALIEN_PLUGIN_TRILINOS, whether [Trilinos][trilinos] plugin is compiled, OFF by
  default
- ALIEN_USE_HDF5, whether HDF5 support is enabled, OFF by default.

We can mention also other generic CMake options :

- CMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed", useful on debian based linux
  distribution (like ubuntu), as without
  linker drops libraries that are not explicitly referenced, breaking our plugin
  interface.
- CMAKE_VERBOSE_MAKEFILE=ON

## Requirements

Alien requires a recent build environment:

- recent CMake (>= 3.18)
- C++ compiler that supports C++-17 (gcc, llvm/clang, intel)
- MPI
- boost, at least with timer and program options components enabled
- glib2
- BLAS
- [Arccon][arcane]
- [Arccore][arcane]
- Google Tests, for unit tests

On Ubuntu-20.04, installing these packages is sufficient for running hypre
solvers:

 ```shell script
apt-get install build-essential cmake gcc g++ gdb \
        libhypre-dev \
        libboost-dev libboost-program-options-dev libgtest-dev libglib2.0-dev
```

For GoogleTest, one must finish installation by running:

```shell script
cd $(mktemp -d) && cmake /usr/src/googletest && cmake --build . --target install
```

## How it works ?

This repository contains the following subdirectories:

- src, the main repository for linear algebra,
- plugins, with different plugins for Alien, to
  call [Ginkgo][ginkgo], [hypre][], [PETSc][petsc],
  or [Trilinos][trilinos] external libraries.

For git developers, Arccore and Arccon dependencies can be built on the fly
by setting `ALIENDEV_EMBEDDED` to `ON`.

## Documentation generation

Documentation is available on [documentation][alien], but can be locally built.

You need a python version 3 with sphinx and breathe modules.

You can easily create a conda environment as following:

```shell script
cd $ALIEN_ROOT/tools/python
conda env create -f alien-env.yml
conda activate alien-env
```

Then the `CMake` flag for documentation has to be activated in the Alien
configuration step

```shell script
cmake -S `pwd`/alien \
      -B `pwd`/build-alien \
      -DALIEN_GENERATE_DOCUMENTATION=ON \
       ....
       
make -C `pwd`/build-alien install
make -C `pwd`/build-alien doc_alien
```

The documentation is generated in :

```shell script
firefox `pwd`/build-alien/alien_doc/index.html
```

[alien]: https://arcaneframework.github.io/alien/

[ginkgo]: https://ginkgo-project.github.io/

[hypre]: https://github.com/hypre-space/hypre

[petsc]: https://petsc.org

[trilinos]: https://trilinos.github.io/

[arcane]: https://arcaneframework.github.io/