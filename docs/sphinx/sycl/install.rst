.. _sycl_install:

===================
How to install SYCL
===================

Installing SYCL
===============

Alien's build system is based on CMake.

Getting the sources
-------------------


.. code-block:: bash

    git clone --recurse-submodules -b stable https://github.com/illuhad/hipSYCL

Configuring
-----------

.. code-block:: bash

    export INSTALL_PREFIX=`pwd`/usr/local/gcc102
    export HIPSYCL_INSTALL_PREFIX=${INSTALL_PATH}/sycl2020
    export HIPSYCL_LLVM_INSTALL_PREFIX=${INSTALL_PATH}
    export HIPSYCL_WITH_ROCM=OFF
    export CC=${GCCCORE_ROOT}/bin/gcc
    export CXX=${GCCCORE_ROOT}/bin/g++
    export HIPSYCL_BASE_CC=gcc
    export HIPSYCL_BASE_CXX=g++
    export hipSYCL_DIR=${INSTALL_PATH}/sycl2020/lib/cmake/hipSYCL
    export HIPSYCL_LLVM_BUILD_DIR=$PWD/llvm
    export HIPSYCL_BUILD_DIR=$PWD/sycl2020/hipSYCL
    mkdir build-hipSYCL
    cmake  -S `pwd`/hipSYCL \
           -B `pwd`/build-hipSYCL \
          -DCMAKE_C_COMPILER=$HIPSYCL_LLVM_INSTALL_PREFIX/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=$HIPSYCL_LLVM_INSTALL_PREFIX/llvm/bin/clang++ \
          -DCMAKE_CXX_FLAGS:STRING='--gcc-toolchain=/work/gratienj/local/expl/eb/centos_7/easybuild/software/Core/GCCcore/10.2.0' \
          -DWITH_CPU_BACKEND=ON \
          -DWITH_CUDA_BACKEND=$HIPSYCL_WITH_CUDA \
          -DWITH_ROCM_BACKEND=$HIPSYCL_WITH_ROCM \
          -DLLVM_DIR=$HIPSYCL_LLVM_INSTALL_PREFIX/llvm/lib/cmake/llvm \
          -DROCM_PATH=$HIPSYCL_INSTALL_PREFIX/rocm \
          -DCUDA_TOOLKIT_ROOT_DIR=$HIPSYCL_LLVM_INSTALL_PREFIX/cuda \
          -DCLANG_EXECUTABLE_PATH=$HIPSYCL_LLVM_INSTALL_PREFIX/llvm/bin/clang++ \
          -DCLANG_INCLUDE_PATH=$LLVM_INCLUDE_PATH \
          -DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL_PREFIX \
          -DROCM_LINK_LINE='-rpath $HIPSYCL_ROCM_LIB_PATH -rpath $HIPSYCL_ROCM_PATH/hsa/lib -L$HIPSYCL_ROCM_LIB_PATH -lhip_hcc -lamd_comgr -lamd_hostcall -lhsa-runtime64 -latmi_runtime -rpath $HIPSYCL_ROCM_PATH/hcc/lib -L$HIPSYCL_ROCM_PATH/hcc/lib -lmcwamp -lhc_am' \


Installing
----------

.. code-block:: bash

    module load GCC/10.2.0
    module load CUDA/10.1
    module load Boost/1.74.0
    module load CMake
    #module load LLVM/11.0.0
    #module load Clang/11.0.1
    
    export CUDA_TOOLKIT_ROOT_DIR= ...
    export CUDA_SDK_ROOT_DIR= ...
    export INSTALL_PATH=`pwd`/Install
    
    export HIPSYCL_PKG_LLVM_VERSION_MAJOR=10
    export INSTALL_PREFIX=`pwd`/usr/local/gcc102
    export HIPSYCL_INSTALL_PREFIX=${INSTALL_PATH}/sycl2020
    export HIPSYCL_LLVM_INSTALL_PREFIX=${INSTALL_PATH}
    export HIPSYCL_WITH_ROCM=OFF
    export CC=${GCCCORE_ROOT}/bin/gcc
    export CXX=${GCCCORE_ROOT}/bin/g++
    export HIPSYCL_BASE_CC=gcc
    export HIPSYCL_BASE_CXX=g++
    export hipSYCL_DIR=${INSTALL_PATH}/sycl2020/lib/cmake/hipSYCL
    export HIPSYCL_LLVM_BUILD_DIR=$PWD/llvm
    export HIPSYCL_BUILD_DIR=$PWD/sycl2020/hipSYCL
    export LD_LIBRARY_PATH=$HIPSYCL_LLVM_INSTALL_PREFIX/llvm/lib:$LD_LIBRARY_PATH

    sh ${ALIEN_ROOT}/tools/sycl/install-llvm.sh
    
    sh ${ALIEN_ROOT}/tools/syclinstall-hipsycl.sh
