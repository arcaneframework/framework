[//]: <> (Comment: -*- coding: utf-8-with-signature -*-)
<img src="https://www.cea.fr/PublishingImages/cea.jpg" height="50" align="right" />
<img src="https://www.ifpenergiesnouvelles.fr/sites/ifpen.fr/files/logo_ifpen_2.jpg" height="50" align="right"/>

Written by CEA/IFPEN and Contributors

(C) Copyright 2000-2024 CEA/IFPEN. All rights reserved.

All content is the property of the respective authors or their employers.

For more information regarding authorship of content, please consult the listed source code repository logs.
____

<p align="center">
  <a href="https://github.com/arcaneframework/framework">
    <img alt="Arcane Framework" src="arcane/doc/theme/img/arcane_framework_medium.webp" width="602px">
  </a>
  <p align="center">Development platform for unstructured 2D or 3D parallel computing codes.</p>
</p>

![GitHub](https://img.shields.io/github/license/arcaneframework/framework?style=for-the-badge)
![GitHub all releases](https://img.shields.io/github/downloads/arcaneframework/framework/total?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/arcaneframework/framework?style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/arcaneframework/framework?style=for-the-badge)
![Codecov](https://img.shields.io/codecov/c/gh/arcaneframework/framework?style=for-the-badge)
![Codacy grade](https://img.shields.io/codacy/grade/9d31bc0a9ae04f858a26342092cb2744?style=for-the-badge)
![Coverity Scan](https://img.shields.io/coverity/scan/24734?style=for-the-badge)

Arcane is a development environment for parallel numerical calculation codes. It supports the architectural aspects of a calculation code, such as data structures for meshing and parallelism, as well as more environment-related aspects such as dataset configuration.
____

Table of Contents:
- [Documentation](#documentation)
  - [User documentation](#user-documentation)
  - [Developer documentation](#developer-documentation)
  - [Alien documentation](#alien-documentation)
- [Changelog](#changelog)
- [Key features](#key-features)
- [Getting started](#getting-started)
  - [Compiling and/or installing Arcane](#compiling-andor-installing-arcane)
    - [Docker images](#docker-images)
    - [Spack](#spack)
- [Examples of how to use the Arcane Framework](#examples-of-how-to-use-the-arcane-framework)
  - [Arcane Benchs](#arcane-benchs)
  - [Sharc](#sharc)
  - [ArcaneFem](#arcanefem)
  - [MaHyCo](#mahyco)
- [Rencontres Arcane](#rencontres-arcanes)
- [Screenshots](#screenshots)

# Documentation

The documentation is available online and is generated and stored in this [GitHub repository](https://github.com/arcaneframework/arcaneframework.github.io).

## User documentation

This documentation is intended for Arcane users.

- [User documentation (French)](https://arcaneframework.github.io/arcane/userdoc/html/index.html)
- User documentation (English) (Soon)

## Developer documentation

This documentation is intended for Arcane developers.

- [Developer documentation (French)](https://arcaneframework.github.io/arcane/devdoc/html/index.html)
- Developer documentation (English) (Soon)

## Alien documentation

This documentation is intended for Alien users and developers.

- [Alien documentation](https://arcaneframework.github.io/framework/aliendoc/html/index.html)

# Changelog

The changelog is available [in the documentation](https://arcaneframework.github.io/arcane/userdoc/html/da/d0c/arcanedoc_news_changelog.html) 
or [in this repository](arcane/doc/doc_common/changelog.md).

# Key features

<details>
  <summary><strong>Massively parallel</strong></summary>

  - Work on simple laptop or on a supercomputer (run case on more than 100k CPU core)
  - Domain partitioning with message-passing (MPI, Shared Memory or hydrid MPI/Shared Memory)
  - Unified Accelerator API (experimental)
    - CUDA
    - ROCM
    - SYCL (experimental)
    - oneTBB (experimental)
  - Basic support for explicit vectorization
  - Automatic Load Balancing with cell migration with several mesh partitioner
    - ParMetis
    - PTScotch
    - Zoltan

</details>

<details>
  <summary><strong>I/O</strong></summary>

  - Supported input mesh file type:
    - VTK Legacy (2.0 and 4.2)
    - VTK VTU
    - [Lima](https://github.com/LIHPC-Computational-Geometry/lima)
    - MED
    - xmf
    - Gmsh (4.1)
  - Post-processing with the following supported output file type:
    - VTKHDF (V1 and V2)
    - Ensight7Gold
  - Time history curves
  - Support for checkpoint/restarting for long simulation

</details>

<details>
  <summary><strong>Mesh entities</strong></summary>

  - Multiple mesh entities (items) are usable in Arcane:
    - Node (0D), Edge (1D), Face and Cells
    - Particle
    - DoF (Degree of freedom)
  - Full Connectivities between items (cell to node, node to edge, ...)
  - Support for user connectivities
  - Easy-to-use Arcane mesh variables on any kind of item
    - double, Int32, Int64, ..
    - Scalar, 1D, 2D or multi-dim
    - and many more...
  - Several kind of meshes are supported (1D, 2D and 3D)
    - unstructured
    - unstructured with adaptative refinement
    - cartesian (experimental)
    - cartesian with patch refinement (AMR) (experimental)
    - polyedral mesh (any number of edges/faces per cell)
  - Meshes are fully dynamic (adding/removing cells)
  - All the connectivities and variables are usable on accelerators

</details>

<details>
  <summary><strong>Multi-constituents support</strong></summary>

  - Two levels of constituents (environment and materials)
  - Any cell variable may have values par constituant

</details>

<details>
  <summary><strong>Performance and Verification and Validation</strong></summary>

  - Bit-to-bit comparison for Arcane variables
  - Between-synchronizations comparing
  - Unit test system integration
  - Automatic profiling of loops using accelerator API
  - Automatic profiling with sampling using Papi library or signal using SIGPROF
  - Automatic use of CUDA CUPTI library to track unified memory (USM) moves between host and device

</details>

<details>
  <summary><strong>Decoupled and extensible framework</strong></summary>

  - Handling of case options via a descriptor file based on XML (axl files)
  - Automatic generation of documentation
  - Notion of Service (plugin) with separated interface/implementation to extend functionalities
  - Notion of independant Modules to enable modularity
  - Limited usage of C++ templates to make extensibility easier

</details>

<details>
  <summary><strong>Extensions to other langages</strong></summary>

  - Most of the classes are available in C#
  - Python wrapping (work in progress, available at end of 2024)

</details>

<details>
  <summary><strong>Algebraic manipulation</strong></summary>

  - Use of Alien library to handle linear systems
  - Coupling with several linear solver libraries
    - Hypre
    - Trilinos
    - PETSc

</details>

<details>
  <summary><strong>Other functionalities</strong></summary>

  - Standalone mode to use only some functionalities of Arcane
    - mesh connectivities
    - accelerator API
  - Handling of time loop with automatic support to go back to a previous iteration

</details>

# Getting started

## Compiling and/or installing Arcane

To compile the Arcane Framework, you need an x86 or ARM64 CPU, Linux OS or Windows OS and a C++
compiler with a version that supports C++20 or higher.

For all other dependencies and more information, check out the 
[Compiling/Installing](https://arcaneframework.github.io/arcane/userdoc/html/d7/d94/arcanedoc_build_install.html)
guide.

<details>
  <summary>Click here if the documentation is not available</summary>

To prepare your computer :

- [Ubuntu 24.04](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/_ubuntu24.md)
- [Ubuntu 22.04](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/_ubuntu22.md) and [CMake instructions](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/snippets/_cmake.md)
- [Ubuntu 20.04](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/_ubuntu20.md) and [CMake instructions](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/snippets/_cmake.md)
- [AlmaLinux/RedHat 9](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/_rh9.md) and [CMake instructions](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/snippets/_cmake.md)
- [ArchLinux based](arcane/doc/doc_common/chap_build_install/subchap_prerequisites/_arch.md)

To build and install Arcane:
- [Compiling/Installing](arcane/doc/doc_common/chap_build_install/2_build.md)
</details>

### Docker images

Docker images with the Arcane framework installed are available in this 
[GitHub repository](https://github.com/arcaneframework/containers/pkgs/container/arcane_ubuntu-2204).
More information [here](https://github.com/arcaneframework/containers).

### Spack

Spack recipes for Arcane are available [here](https://github.com/arcaneframework/spack_recipes).

# Examples of how to use the Arcane Framework

An introductory chapter with the construction of a Hello world is
available [here](https://arcaneframework.github.io/arcane/userdoc/html/db/d53/arcanedoc_examples.html).

Examples of applications using Arcane are available on GitHub. 
Here is a non-exhaustive list:

## [Arcane Benchs](https://github.com/arcaneframework/arcane-benchs)

A set of mini-applications to evaluate Arcane functionalities. These are a good basis for getting 
started with Arcane.

## [Sharc](https://github.com/arcaneframework/sharc)

Arcane-based application for solving different geosciences problems.

## [ArcaneFem](https://github.com/arcaneframework/arcanefem)

Very simple codes to test Finite Element Methods using Arcane.

## [MaHyCo](https://github.com/cea-hpc/MaHyCo)

Finite volume code for solving hydrodynamic equations:
Lagrangian or Eulerian simulations.

# Rencontres Arcane

The next Rencontres Arcane are scheduled on Monday, the 24th March, 2025.

<p align="center">
  <a href="https://github.com/arcaneframework/events">
    <img alt="Arcane Framework" src="https://raw.githubusercontent.com/arcaneframework/events/main/rencontresarcane2025/visuel/BandeauARCANE_2025_V2.png" width="700px">
  </a>
  <p align="center">Les Rencontres Arcane on March 24th 2025.</p>
</p>

The previous presentations are stored [here](https://github.com/arcaneframework/events).

# Screenshots

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/tree/main/elastodynamics">
    <img alt="Transient elastodynamics" src="https://github.com/arcaneframework/arcanefem/assets/52162083/692ba9e7-5dbd-450a-ab19-e6c4a0df58a6" width="600px">
  </a>
  <p align="center">Transient elastodynamics with ArcaneFEM</p>
</p>


<details>
  <summary><strong>More Screenshots from ArcaneFEM</strong></summary>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/tree/main/aerodynamics">
    <img alt="Aerodynamics" src="https://github.com/arcaneframework/arcanefem/assets/52162083/8c691cee-d8e8-463a-b9b1-c00d016386f5" width="600px">
  </a>
  <p align="center">Aerodynamics</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/bilaplacian">
    <img alt="Bilaplacian" src="https://github.com/arcaneframework/arcanefem/assets/52162083/9f183f44-cc7c-40cb-9b6b-8fefdf0f94bf" width="400px">
  </a>
  <p align="center">Bilaplacian</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/elasticity">
    <img alt="Linear elasticity" src="https://github.com/arcaneframework/arcanefem/assets/52162083/eb970ece-5fd3-4862-9b93-e8930a103ae9" width="500px">
  </a>
  <p align="center">Linear elasticity</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/electrostatics">
    <img alt="Electrostatics" src="https://github.com/arcaneframework/arcanefem/assets/52162083/959988a3-1717-4449-b412-14cbd1582367" width="500px">
  </a>
  <p align="center">Electrostatics</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/fourier">
    <img alt="Solving Fourier equation" src="https://github.com/arcaneframework/arcanefem/assets/52162083/cf86f60f-360f-491b-a234-9631fc27af45" width="600px">
  </a>
  <p align="center">Solving Fourier equation</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/laplace">
    <img alt="Solving Laplace equation with ArcaneFEM" src="https://github.com/arcaneframework/arcanefem/assets/52162083/be3d2ea6-bfb7-42d9-b82e-a62509a498f8" width="400px">
  </a>
  <p align="center">Solving Laplace equation</p>
</p>

<p align="center">
  <a href="https://github.com/arcaneframework/arcanefem/blob/main/poisson">
    <img alt="Solving Poisson equation with ArcaneFEM" src="https://github.com/arcaneframework/arcanefem/assets/52162083/a8d114e1-5589-4efd-88fd-84b398acab84" width="400px">
  </a>
  <p align="center">Solving Poisson equation</p>
</p>

</details>


<p align="center">
  <a href="https://github.com/arcaneframework/sharc">
    <img alt="t=28" src="https://raw.githubusercontent.com/arcaneframework/resources/master/screenshots/sharc/conc28.jpeg" width="400px">
  </a>
  <p align="center">Water concentration in a porous material with Sharc</p>
</p>

<details>
  <summary><strong>More Screenshots from Sharc</strong></summary>

<p align="center">
  <a href="https://github.com/arcaneframework/sharc">
    <img alt="t=34" src="https://raw.githubusercontent.com/arcaneframework/resources/master/screenshots/sharc/conc34.jpeg" width="400px">
  </a>

  <a href="https://github.com/arcaneframework/sharc">
    <img alt="t=40" src="https://raw.githubusercontent.com/arcaneframework/resources/master/screenshots/sharc/conc40.jpeg" width="400px">
  </a>
  <p align="center">Evolution of water concentration over time in a porous material</p>
</p>

</details>