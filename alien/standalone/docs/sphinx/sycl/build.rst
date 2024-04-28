.. _sycl_build:

=====================================
Building and using Alien SYCL backend
=====================================

Building Alien SYCL backend
===========================

Alien's build system is based on CMake.


Configuring with Gcc compiler, hipSYCL with cuda support
--------------------------------------------------------


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

Configuring with Clang compiler, hipSYCL with hip-rocm support
--------------------------------------------------------------

.. code-block:: bash

    export ROCM_ROOT=/opt/rocm-5.5.1
    export CC=$ROCM_ROOT/llvm/bin/clang
    export CXX=$ROCM_ROOT/llvm/bin/clang++
    export ROOT_DIR=/lus/work/CT2A/cad14948/SHARED
    export PREFIX_PATH="$ROCM_ROOT;$ROCM_ROOT/hip"
    export HIP_ARCHITECTURES=gfx90a    # AMD Instinct MI300 = gfx940 architecture
    cmake -S `pwd`/framework \
          -B `pwd`/build  \
          -DCMAKE_BUILD_TYPE=${BuildType} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
          -DARCANE_WANT_ARCCON_EXPORT_TARGET=OFF \
          -DFRAMEWORK_NO_EXPORT_PACKAGES=ON \
          -DALIEN_BUILD_COMPONENT=all \
          -DALIEN_USE_EIGEN3=ON \
          -DALIEN_USE_SYCL=ON \
          -DALIEN_USE_HIPSYCL=ON \
          -DHIPSYCL_TARGETS=hip:gfx90a \
          -DSYCL_INCLUDE_DIR_HINT=${HIPSYCL_ROOT}/include \
          -DARCANE_ACCELERATOR_MODE=ROCMHIP \
          -DCMAKE_HIP_ARCHITECTURES=${HIP_ARCHITECTURES} \
          -DARCCORE_CXX_STANDARD=20 \
          -DCMAKE_AMDGPU_TARGETS=gfx90a \
          -DCMAKE_GPU_TARGETS=gfx90a
          

Configuring with oneAPI ipcx compiler
-------------------------------------


.. code-block:: bash

    export CXX=icpx
    export CC=icx
    cmake -S `pwd`/framework \
          -B `pwd`/build${SUF}  \
          -DCMAKE_BUILD_TYPE=${BuildType} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
          -DARCANE_WANT_ARCCON_EXPORT_TARGET=OFF \
          -DFRAMEWORK_NO_EXPORT_PACKAGES=ON \
          -DALIEN_BUILD_COMPONENT=all \
          -DALIEN_USE_EIGEN3=ON \
          -DALIEN_USE_SYCL=0N \
          -DALIEN_USE_INTELSYCL=ON \
          -DALIEN_USE_CUDA=ON \
          -DONEAPI_CXX_COMPILER=${CXX} \
          -DARCCORE_CXX_STANDARD=20 \
          -DARCANE_ACCELERATOR_MODE=CUDANVCC \
          -DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc \
          -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

Concepts
--------

