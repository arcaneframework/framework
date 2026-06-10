// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonCudaHipAtomicImpl.h                                   (C) 2000-2026 */
/*                                                                           */
/* CUDA and HIP implementation of atomic operations.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_COMMONCUDHIPATOMICIMPL_H
#define ARCCORE_ACCELERATOR_COMMONCUDHIPATOMICIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// This file must only be included by 'arcane/accelerator/Reduce.h'
// and is only valid when compiled by the CUDA and HIP compilers

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Warning: with ROCm and a GPU on a PCI express bus, most
// atomic methods do not work if the pointer is allocated
// in unified memory. The problem seems to occur with atomicMin, atomicMax,
// atomicInc. However, atomicAdd seems to work if concurrent accesses
// are not too numerous.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, enum eAtomicOperation>
class CommonCudaHipAtomic;

template <typename DataType>
class CommonCudaHipAtomicAdd;
template <typename DataType>
class CommonCudaHipAtomicMax;
template <typename DataType>
class CommonCudaHipAtomicMin;

template <>
class CommonCudaHipAtomic<int, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE int apply(int* ptr, int v)
  {
    return ::atomicAdd(ptr, v);
  }
};

template <>
class CommonCudaHipAtomic<int, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE int apply(int* ptr, int v)
  {
    return ::atomicMax(ptr, v);
  }
};

template <>
class CommonCudaHipAtomic<int, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE int apply(int* ptr, int v)
  {
    return ::atomicMin(ptr, v);
  }
};

template <>
class CommonCudaHipAtomic<Int64, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE Int64 apply(Int64* ptr, Int64 v)
  {
    static_assert(sizeof(Int64) == sizeof(long long int), "Bad pointer size");
    return static_cast<Int64>(::atomicAdd((unsigned long long int*)ptr, v));
  }
};

template <>
class CommonCudaHipAtomic<Int64, eAtomicOperation::Max>
{
 public:

#if defined(__HIP__)
  static ARCCORE_DEVICE Int64 apply(Int64* ptr, Int64 v)
  {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old = *address_as_ull, assumed;

    do {
      assumed = old;
      Int64 assumed_as_int64 = static_cast<Int64>(assumed);
      old = atomicCAS(address_as_ull, assumed,
                      static_cast<unsigned long long int>(v > assumed_as_int64 ? v : assumed_as_int64));
    } while (assumed != old);
    return static_cast<Int64>(old);
  }
#else
  static ARCCORE_DEVICE Int64 apply(Int64* ptr, Int64 v)
  {
    return static_cast<Int64>(::atomicMax((long long int*)ptr, v));
  }
#endif
};

template <>
class CommonCudaHipAtomic<Int64, eAtomicOperation::Min>
{
 public:

#if defined(__HIP__)
  static ARCCORE_DEVICE Int64 apply(Int64* ptr, Int64 v)
  {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old = *address_as_ull, assumed;

    do {
      assumed = old;
      Int64 assumed_as_int64 = static_cast<Int64>(assumed);
      old = atomicCAS(address_as_ull, assumed,
                      static_cast<unsigned long long int>(v < assumed_as_int64 ? v : assumed_as_int64));
    } while (assumed != old);
    return static_cast<Int64>(old);
  }
#else
  static ARCCORE_DEVICE Int64 apply(Int64* ptr, Int64 v)
  {
    return static_cast<Int64>(::atomicMin((long long int*)ptr, v));
  }
#endif
};

// Devices with architecture less than 6.0 do not support
// atomicAdd on 'double'.
// This code is derived from NVIDIA documentation (programming guide)
__device__ inline double
preArch60atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
__device__ inline double
atomicMaxDouble(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    double assumed_as_double = __longlong_as_double(assumed);
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val > assumed_as_double ? val : assumed_as_double));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ inline double
atomicMinDouble(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    double assumed_as_double = __longlong_as_double(assumed);
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val < assumed_as_double ? val : assumed_as_double));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

template <>
class CommonCudaHipAtomic<double, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE double apply(double* ptr, double v)
  {
#if __CUDA_ARCH__ >= 600
    return ::atomicAdd(ptr, v);
#else
    return preArch60atomicAdd(ptr, v);
#endif
  }
};

template <>
class CommonCudaHipAtomic<double, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE double apply(double* ptr, double v)
  {
    return atomicMaxDouble(ptr, v);
  }
};

template <>
class CommonCudaHipAtomic<double, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE double apply(double* ptr, double v)
  {
    return atomicMinDouble(ptr, v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
