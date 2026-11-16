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

// This file must only be included by 'arcane/accelerator/Atomic.h'
// and is only valid when compiled by the CUDA and HIP compilers

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Macro to directly use CUDA or ROCM atomic operation
#define ARCCORE_IMPL_USE_CUDAROCM_ATOMIC(datatype) \
  template <> \
  class CommonCudaHipAtomic<datatype, eAtomicOperation::Add> \
  { \
   public: \
\
    static ARCCORE_DEVICE datatype apply(datatype* ptr, datatype v) \
    { \
      return ::atomicAdd(ptr, v); \
    } \
  }; \
  template <> \
  class CommonCudaHipAtomic<datatype, eAtomicOperation::Max> \
  { \
   public: \
\
    static ARCCORE_DEVICE datatype apply(datatype* ptr, datatype v) \
    { \
      return ::atomicMax(ptr, v); \
    } \
  }; \
  template <> \
  class CommonCudaHipAtomic<datatype, eAtomicOperation::Min> \
  { \
   public: \
\
    static ARCCORE_DEVICE datatype apply(datatype* ptr, datatype v) \
    { \
      return ::atomicMin(ptr, v); \
    } \
  };

ARCCORE_IMPL_USE_CUDAROCM_ATOMIC(int);
ARCCORE_IMPL_USE_CUDAROCM_ATOMIC(unsigned int);
ARCCORE_IMPL_USE_CUDAROCM_ATOMIC(unsigned long long);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class CommonCudaHipAtomic<long long, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE long long apply(long long* ptr, long long v)
  {
    void* p = ptr;
    return static_cast<long long>(::atomicAdd(static_cast<unsigned long long*>(p), v));
  }
};

template <>
class CommonCudaHipAtomic<long long, eAtomicOperation::Max>
{
 public:

#if defined(__HIP__)
  static ARCCORE_DEVICE long long apply(long long* ptr, long long v)
  {
    void* p = ptr;
    unsigned long long int* address_as_ull = static_cast<unsigned long long int*>(p);
    unsigned long long int old = *address_as_ull, assumed;

    do {
      assumed = old;
      long long assumed_as_int64 = static_cast<Int64>(assumed);
      old = atomicCAS(address_as_ull, assumed,
                      static_cast<unsigned long long int>(v > assumed_as_int64 ? v : assumed_as_int64));
    } while (assumed != old);
    return static_cast<long long>(old);
  }
#else
  static ARCCORE_DEVICE long long apply(long long* ptr, long long v)
  {
    void* p = ptr;
    return static_cast<long long>(::atomicMax(static_cast<long long int*>(p), v));
  }
#endif
};

template <>
class CommonCudaHipAtomic<long long, eAtomicOperation::Min>
{
 public:

#if defined(__HIP__)
  static ARCCORE_DEVICE long long apply(long long* ptr, long long v)
  {
    void* p = ptr;
    unsigned long long int* address_as_ull = static_cast<unsigned long long int*>(p);
    unsigned long long int old = *address_as_ull, assumed;

    do {
      assumed = old;
      long long assumed_as_int64 = static_cast<Int64>(assumed);
      old = atomicCAS(address_as_ull, assumed,
                      static_cast<unsigned long long int>(v < assumed_as_int64 ? v : assumed_as_int64));
    } while (assumed != old);
    return static_cast<long long>(old);
  }
#else
  static ARCCORE_DEVICE long long apply(long long* ptr, long long v)
  {
    void* p = ptr;
    return static_cast<long long>(::atomicMin(static_cast<long long int*>(p), v));
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class CommonCudaHipAtomic<long, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE long apply(long* ptr, long v)
  {
    void* p = ptr;
    if constexpr (sizeof(long) == sizeof(long long))
      return static_cast<long>(::atomicAdd(static_cast<unsigned long long int*>(p), v));
    else
      return ::atomicAdd(static_cast<int*>(p), v);
  }
};

template <>
class CommonCudaHipAtomic<long, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE long apply(long* ptr, long v)
  {
    void* p = ptr;
    if constexpr (sizeof(long) == sizeof(long long))
      return CommonCudaHipAtomic<long long, eAtomicOperation::Max>::apply(static_cast<long long*>(p), v);
    else
      return ::atomicMax(static_cast<int*>(p), v);
  }
};

template <>
class CommonCudaHipAtomic<long, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE long apply(long* ptr, long v)
  {
    void* p = ptr;
    if constexpr (sizeof(long) == sizeof(long long))
      return CommonCudaHipAtomic<long long, eAtomicOperation::Min>::apply(static_cast<long long*>(p), v);
    else
      return ::atomicMin(static_cast<int*>(p), v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class CommonCudaHipAtomic<unsigned long, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE unsigned long apply(unsigned long* ptr, unsigned long v)
  {
    void* p = ptr;
    if constexpr (sizeof(unsigned long) == sizeof(long long))
      return static_cast<unsigned long>(::atomicAdd(static_cast<unsigned long long*>(p), v));
    else
      return ::atomicAdd(static_cast<unsigned int*>(p), v);
  }
};

template <>
class CommonCudaHipAtomic<unsigned long, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE unsigned long apply(unsigned long* ptr, unsigned long v)
  {
    void* p = ptr;
    if constexpr (sizeof(unsigned long) == sizeof(long long))
      return CommonCudaHipAtomic<unsigned long long, eAtomicOperation::Max>::apply(static_cast<unsigned long long*>(p), v);
    else
      return ::atomicMax(static_cast<unsigned int*>(p), v);
  }
};

template <>
class CommonCudaHipAtomic<unsigned long, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE long apply(unsigned long* ptr, unsigned long v)
  {
    void* p = ptr;
    if constexpr (sizeof(unsigned long) == sizeof(unsigned long long))
      return CommonCudaHipAtomic<unsigned long long, eAtomicOperation::Min>::apply(static_cast<unsigned long long*>(p), v);
    else
      return ::atomicMin(static_cast<unsigned int*>(p), v);
  }
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
// AtomicMax and AtomicMix is not supported for CUDA on 'float'.

__device__ inline float
atomicMaxFloat(float* address, float val)
{
  static_assert(sizeof(float) == sizeof(unsigned int), "Bad pointer size");
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned long long int old = *address_as_ui, assumed;

  do {
    assumed = old;
    float assumed_as_float = __uint_as_float(assumed);
    old = atomicCAS(address_as_ui, assumed,
                    __float_as_uint(val > assumed_as_float ? val : assumed_as_float));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

__device__ inline float
atomicMinFloat(float* address, float val)
{
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui, assumed;

  do {
    assumed = old;
    float assumed_as_float = __uint_as_float(assumed);
    old = atomicCAS(address_as_ui, assumed,
                    __float_as_uint(val < assumed_as_float ? val : assumed_as_float));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

template <>
class CommonCudaHipAtomic<float, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE float apply(float* ptr, float v)
  {
    return ::atomicAdd(ptr, v);
  }
};

template <>
class CommonCudaHipAtomic<float, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE float apply(float* ptr, float v)
  {
#if defined(__HIP__)
    return ::atomicMax(ptr, v);
#else
    return atomicMaxFloat(ptr, v);
#endif
  }
};

template <>
class CommonCudaHipAtomic<float, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE float apply(float* ptr, float v)
  {
#if defined(__HIP__)
    return ::atomicMin(ptr, v);
#else
    return atomicMinFloat(ptr, v);
#endif
  }
};

#undef ARCCORE_IMPL_USE_CUDAROCM_ATOMIC

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
