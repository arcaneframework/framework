.. _user_build:

========================
Building and Using Alien
========================

Building Alien
==============

Alien's build system is based on CMake.

Getting the sources
-------------------

One can use Alien's github repository https://github.com/arcaneframework/alien
to either get an archive of a released version https://github.com/arcaneframework/alien/releases or
to clone the repository.

Configuring
-----------

When using a released version, Arccon and Arccore must be installed before being able to compile Alien.
Their install paths can be passed to cmake using `CMAKE_PREFIX_PATH`.

Optional dependencies are:
 - HDF5, enabled by setting `ALIEN_USE_HDF5` to `ON`
 - libxml2, enabled by setting `ALIEN_USE_xml2` to `ON`
 - GoogleTest, enabled by setting `ALIEN_UNIT_TESTS` to `ON`

`ALIEN_DEFAULT_OPTIONS` allows to try detecting installed dependencies and automatically enable them.

Backend librairies are:
 - hypre, enabled by setting `ALIEN_PLUGIN_HYPRE` to `ON`
 - PETSc, enabled by setting `ALIEN_PLUGIN_PETSC` to `ON`

Example
-------

Configuring, compiling Alien, using hypre and petsc.

.. code-block:: bash

    cmake -DALIEN_DEFAULT_OPTIONS:BOOL=ON -DALIEN_PLUGIN_HYPRE:BOOL=ON -DALIEN_PLUGIN_PETSC:BOOL=ON -B <build_dir> <alien_src_path>
    cmake --build <build_dir>
    cmake --install <build_dir>

Using Alien from a CMake project
================================

Example using hypre and move semantic:

.. code-block:: cmake

    find_package(Alien REQUIRED)

    add_library(foo <your_src>)
    target_link_libraries(foo PRIVATE Alien::hypre_wrapper Alien::semantic_move)


