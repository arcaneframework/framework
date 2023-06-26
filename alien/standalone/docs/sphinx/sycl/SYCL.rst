.. _sycl_SYCL:

=====================
SYCL backend in ALIEN
=====================

Introduction
============

Alien provide a SYCL backend to handle NVidia, AMD and Intel GP-GPUs

It is based on the hipSYCL implementation of the SYCL 2020 API.

It depends on :

- LLVM and Clang
- CUDA 10 to handle NVidia GP-GPUs
- ROCM to handle AMD GP-GPUs
- OneAPI and DPC++ for Intel GP-GPUs

It provides a Block EllPack Matrix implementation and a Linear Algebra with all 
the Blas 1 and 2 operations required to implement the CG and BiCGStab krylov algorithms. 