# ALIEN TOOL SET

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2024 CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult 
the listed source code repository logs.

## Introduction

Alien is a set a tools to handle and solve Linear Systems in applications.

Alien is composed of three main sub-projets:

- standalone : provides the core library
- plugins : provides plugins of external solver package like hypre, petsc, 
  trilinos, ginkgo
- ArcaneInterface : provides IFPEN legacy plugins 
  (hypre, petsc, mtl4, trilinos, hpddm, ifpsolver, mcgsolver and htssolver) 
  and their adapters to Arcane services and 
  a set of tools to manage the links between 
  Arcane variables and mesh entities with algebraic indexes.


## Installation

Simply clone the framework repository.

```shell script
git clone https://github.com/arcaneframework/framework.git
cd alien
```


After, you can run CMake (at least version 3.18).

```shell script
mkdir build && cd build
cmake ..
```

Useful CMake options:

- ALIEN_BUILD_COMPONENT=all, compile all components 
- ALIEN_PLUGIN_PETSC=ON,OFF enable compile [Petsc][petsc] plugin default=OFF
- ALIEN_PLUGIN_HYPRE=ON,OFF enable compile [Hypre][hypre] plugin default=OFF
- ALIEN_PLUGIN_TRILINOS=ON,OFF enable [Trilinos][trilinos] plugin default=OFF
- ALIEN_PLUGIN_GINKGO=ON,OFF enable [Ginkgo][ginkgo] plugin default=OFF
- ALIEN_USE_HDF5=ON,OFF enable HDF5 support for import-export tools
- ALIEN_USE_LIBXML2=ON,OFF enable LIBXML2 support for import-export tool
- ALIEN_USE_EIGEN3=ON,OFF enable eigen3 support
- ALIEN_GENERATE_DOCUMENTATION=ON,OFF enable doc generation, default=OFF
- ALIEN_USE_SYCL=ON,OFF enable SYCL backend support, default=OFF
- ALIEN_USE_CUDA=ON,OFF enable SYCL backend with CUDA, default=OFF
- ALIEN_USE_HIP=ON,OFF enable  SYCL backend with HIP default=OFF
- ALIEN_USE_HIPSYCL=ON,OFF to use HIPSYCL to compile SYCL backend, default=OFF
- ALIEN_USE_INTELSYCL=ON,OFF to use OneAPI icpx compiler for SYCL backend



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
cmake -S `pwd`/framework \
      -B `pwd`/build \
      -DALIEN_GENERATE_DOCUMENTATION=ON \
       ....
       
make -C `pwd`/build install
make -C `pwd`/build docalien
```

The documentation is generated in :

```shell script
firefox `pwd`/build-alien/alien_doc/index.html
```

[alien]: https://arcaneframework.github.io/framework/aliendoc/html/index.html

[ginkgo]: https://ginkgo-project.github.io/

[hypre]: https://github.com/hypre-space/hypre

[petsc]: https://petsc.org

[trilinos]: https://trilinos.github.io/

[arcane]: https://arcaneframework.github.io/
