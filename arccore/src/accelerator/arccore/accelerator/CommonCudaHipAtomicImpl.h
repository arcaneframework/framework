// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonCudaHipAtomicImpl.h                                   (C) 2000-2025 */
/*                                                                           */
/* Implémentation CUDA et HIP des opérations atomiques.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_COMMONCUDHIPATOMICIMPL_H
#define ARCCORE_ACCELERATOR_COMMONCUDHIPATOMICIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Ce fichier doit être inclus uniquement par 'arcane/accelerator/Reduce.h'
// et n'est valide que compilé par le compilateur CUDA et HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention: avec ROCm et un GPU sur bus PCI express la plupart des
// méthodes atomiques ne fonctionnent pas si le pointeur est allouée
// en mémoire unifiée. A priori le problème se pose avec atomicMin, atomicMax,
// atomicInc. Par contre atomicAdd a l'air de fonctionner si les accès
// concurrents ne sont pas trop nombreux.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
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

// Les devices d'architecture inférieure à 6.0 ne supportent pas
// les atomicAdd sur les 'double'.
// Ce code est issu de la documentation NVIDIA (programming guide)
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

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
