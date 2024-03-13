.. _sycl_SYCL:

=====================
SYCL backend in ALIEN
=====================

Introduction
============

Alien provide a SYCL backend to handle NVidia, AMD and Intel GP-GPUs

Alien SYCL backend has been tested with:

- AdaptiveCPP (former hipSYCL) implementation of the SYCL 2020 API version 0.9.4 with CUDA and HIP-ROCM
- oneAPI 2024.0, dppc++

It depends on :

- LLVM and Clang
- CUDA 12 to handle NVidia GP-GPUs
- ROCM to handle AMD GP-GPUs
- OneAPI and DPC++ for Intel GP-GPUs

It provides a Block EllPack Matrix implementation and a Linear Algebra with all 
the Blas 1 and 2 operations required to implement the CG and BiCGStab krylov algorithms. 

Some Matrix Vector Builders abd Accessors are provided to enable Matrix and Vector assembly directly on the device memory.