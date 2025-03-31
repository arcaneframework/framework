.. _sycl_install:

===================
How to install SYCL
===================

Installing OneAPI 2024.0 with CUDA Support
==========================================

.. code-block:: bash
    //Download oneapi offline 
    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh 
    //Install in silent mode in a non-standard folder
    chmod +x l_BaseKit_p_2024.0.1.46_offline.sh
    ./l_BaseKit_p_2024.0.1.46_offline.sh  -a -s --eula accept --download-cache /temp_path --install-dir /path/to/intel/oneapi 
    //Download plugin for nvidia
    curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2024.0.1&filters[]=12.0&filters[]=linux"
    //Install in the oneapi folder
    ./oneapi-for-nvidia-gpus-2024.0.1-cuda-12.0-linux.sh -y --extract-folder --install-dir /path/to/intel/oneapi


Installing hipSYCL v0.9.4 with CUDA or ROCM Support
===================================================

Alien's build system is based on CMake.

Getting the sources
-------------------


.. code-block:: bash

    git clone --recurse-submodules -b stable https://github.com/illuhad/hipSYCL

Example of configuration hipSYCL with GCC 11 and CUDA 11 ou 12
----------------------------------------------------------

.. code-block:: bash

    export INSTALL_PREFIX=`pwd`/install
    export HIPSYCL_INSTALL_PREFIX=${INSTALL_PREFIX}/sycl2020
    export CLANG_INSTALL_PREFIX=...
    export CUDA_INSTALL_PREFIX=$CUDA_ROOT
    export HIPSYCL_WITH_CUDA=ON
    export HIPSYCL_WITH_ROCM=OFF
    export CC=${GCCCORE_ROOT}/bin/gcc
    export CXX=${GCCCORE_ROOT}/bin/g++
    export HIPSYCL_BASE_CC=gcc
    export HIPSYCL_BASE_CXX=g++
    export hipSYCL_DIR=${INSTALL_PATH}/sycl2020/lib/cmake/hipSYCL
    mkdir build-hipSYCL
    cmake  -S `pwd`/hipSYCL \
           -B `pwd`/build-hipSYCL \
          -DCMAKE_C_COMPILER=$CLANG_INSTALL_PREFIX/bin/clang \
          -DCMAKE_CXX_COMPILER=$CLANG_INSTALL_PREFIX/bin/clang++ \
          -DCMAKE_CXX_FLAGS:STRING='--gcc-toolchain=/work/gratienj/local/expl/eb/centos_7/easybuild/software/Core/GCCcore/10.2.0' \
          -DWITH_CPU_BACKEND=ON \
          -DWITH_CUDA_BACKEND=$HIPSYCL_WITH_CUDA \
          -DWITH_ROCM_BACKEND=$HIPSYCL_WITH_ROCM \
          -DLLVM_DIR=$CLANG_INSTALL_PREFIX/lib/cmake/llvm \
          -DROCM_PATH=$ROCM_INSTALL_PREFIX/rocm \
          -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_INSTALL_PREFIX/cuda \
          -DCLANG_EXECUTABLE_PATH=$CLANG_INSTALL_PREFIX/bin/clang++ \
          -DCLANG_INCLUDE_PATH=$LLVM_INCLUDE_PATH \
          -DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL_PREFIX \
          -DROCM_LINK_LINE='-rpath $HIPSYCL_ROCM_LIB_PATH -rpath $HIPSYCL_ROCM_PATH/hsa/lib -L$HIPSYCL_ROCM_LIB_PATH -lhip_hcc -lamd_comgr -lamd_hostcall -lhsa-runtime64 -latmi_runtime -rpath $HIPSYCL_ROCM_PATH/hcc/lib -L$HIPSYCL_ROCM_PATH/hcc/lib -lmcwamp -lhc_am' \


Example of configuration hipSYCL with Clang and ROCM 5.5.1
----------------------------------------------------------

.. code-block:: bash
 
    export ROCM_ROOT=/opt/rocm-5.5.1
    export LLVM_DIR=/opt/rocm-5.5.1/llvm/lib/cmake/llvm
    export CC=$ROCM_ROOT/llvm/bin/clang
    export CXX=$ROCM_ROOT/llvm/bin/clang++
    export BOOST_ROOT=/opt/software/gaia/prod/1.1.1/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_plac/boost-1.81.0-rocmcc-5.3.0-cky6

    export CC=clang
    export CXX=clang++

    export ROOT_DIR=/lus/work/CT2A/cad14948/SHARED
    export PREFIX_PATH="$ROCM_ROOT;$ROCM_ROOT/hip"

    export HIP_ARCHITECTURES=gfx90a    # AMD Instinct MI300 = gfx940 architecture

    cd buildAdaptiveCPP23
    cmake -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DLLVM_DIR=$ROCM_ROOT/llvm/lib/cmake/llvm \
      -DCLANG_EXECUTABLE_PATH=$ROCM_ROOT/llvm/bin/clang++ \
      -DCLANG_INCLUDE_PATH=$ROCM_ROOT/llvm/include \
      -DROCM_PATH=${ROCM_ROOT} \
      -DWITH_CPU_BACKEND=ON \
      -DWITH_ROCM_BACKEND=ON \
      -WITH_OPENCL_BACKEND=OFF \
      -DWITH_LEVEL_ZERO_BACKEND=OFF \
      -WITH_SSCP_COMPILER=OFF \
      -DCMAKE_INSTALL_PREFIX=/lus/work/CT2A/cad14948/SHARED/local/adaptivecpp/v23.10.0 \
      -DWITH_ACCELERATED_CPU=OFF \
      -DBOOST_ROOT=$BOOST_ROOT \
      /lus/work/CT2A/cad14948/SHARED/AdaptiveCpp-v23.10.0
    make install



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
