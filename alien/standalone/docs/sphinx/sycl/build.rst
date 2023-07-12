.. _sycl_build:

=====================================
Building and using Alien SYCL backend
=====================================

Building Alien SYCL backend
===========================

Alien's build system is based on CMake.


Configuring
-----------


.. code-block:: bash

    export GCCCORE='path_to_gccroot_compiler'
    cmake -S `pwd`/alien \
          -B `pwd`/build-alien \
          -DCMAKE_BUILD_TYPE=Release \
          -DALIEN_WANT_AVX2=ON \
          -DALIEN_USE_LIBXML2=ON \
          -DALIEN_UNIT_TESTS=ON \
          -DALIEN_USE_HDF5=ON -DHIPSYCL_TARGETS=cuda:sm_50 \
          -DALIEN_USE_SYCL=ON \
          -DGCCCORE_ROOT:PATH=${GCCXORE_ROOT} \
          ../alien
    cmake --build <build_dir>
    cmake --install <build_dir>


Concepts
--------

