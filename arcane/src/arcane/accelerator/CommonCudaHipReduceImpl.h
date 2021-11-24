// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonCudaHipReduceImpl.h                                   (C) 2000-2021 */
/*                                                                           */
/* Implémentation CUDA et HIP des réductions.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_COMMONCUDHIPAREDUCEIMPL_H
#define ARCANE_ACCELERATOR_COMMONCUDHIPAREDUCEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Ce fichier doit être inclus uniquement par 'arcane/accelerator/Reduce.h'
// et n'est valide que compilé par le compilateur CUDA et HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
__device__ __forceinline__ unsigned int getThreadId()
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;
  return threadId;
}

template<typename DataType>
class CommonCudaHipAtomicAdd;
template<typename DataType>
class CommonCudaHipAtomicMax;
template<typename DataType>
class CommonCudaHipAtomicMin;

template<>
class CommonCudaHipAtomicAdd<int>
{
 public:
  static ARCCORE_DEVICE void apply(int* ptr,int v)
  {
    ::atomicAdd(ptr,v);
  }
};

template<>
class CommonCudaHipAtomicMax<int>
{
 public:
  static ARCCORE_DEVICE void apply(int* ptr,int v)
  {
    ::atomicMax(ptr,v);
  }
};

template<>
class CommonCudaHipAtomicMin<int>
{
 public:
  static ARCCORE_DEVICE void apply(int* ptr,int v)
  {
    ::atomicMin(ptr,v);
  }
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
                    __double_as_longlong(val > assumed_as_double ? val :assumed_as_double ));
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

template<>
class CommonCudaHipAtomicAdd<double>
{
 public:
  static ARCCORE_DEVICE void apply(double* ptr,double v)
  {
#if __CUDA_ARCH__ >= 600
    ::atomicAdd(ptr,v);
#else
    preArch60atomicAdd(ptr,v);
#endif
  }
};

template<>
class CommonCudaHipAtomicMax<double>
{
 public:
  static ARCCORE_DEVICE void apply(double* ptr,double v)
  {
    atomicMaxDouble(ptr,v);
  }
};

template<>
class CommonCudaHipAtomicMin<double>
{
 public:
  static ARCCORE_DEVICE void apply(double* ptr,double v)
  {
    atomicMinDouble(ptr,v);
  }
};

constexpr const Int32 WARP_SIZE = 32;
constexpr const Int32 MAX_BLOCK_SIZE = 1024;
constexpr const Int32 MAX_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;

template <typename T>
struct SumReduceOperator
{
  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    CommonCudaHipAtomicAdd<T>::apply(&val, v);
  }
};

template <typename T>
struct SimpleSumReduceOperator
{
  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = val + v;
  }
};

template <typename T>
struct SimpleMaxReduceOperator
{
  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = v>val ? v : val;
  }
};

template <typename T>
struct SimpleMinReduceOperator
{
  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = v<val ? v : val;
  }
};

#if defined(__CUDACC__)
ARCCORE_DEVICE inline double shfl_xor_sync(double var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

ARCCORE_DEVICE inline int shfl_xor_sync(int var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

ARCCORE_DEVICE inline double shfl_sync(double var, int laneMask)
{
  return ::__shfl_sync(0xffffffffu, var, laneMask);
}

ARCCORE_DEVICE inline int shfl_sync(int var, int laneMask)
{
  return ::__shfl_sync(0xffffffffu, var, laneMask);
}
#endif
#if defined(__HIP__)
ARCCORE_DEVICE inline double shfl_xor_sync(double var, int laneMask)
{
  return ::__shfl_xor(var, laneMask);
}

ARCCORE_DEVICE inline int shfl_xor_sync(int var, int laneMask)
{
  return ::__shfl_xor(var, laneMask);
}

ARCCORE_DEVICE inline double shfl_sync(double var, int laneMask)
{
  return ::__shfl(var, laneMask);
}

ARCCORE_DEVICE inline int shfl_sync(int var, int laneMask)
{
  return ::__shfl(var, laneMask);
}
#endif

// Cette implémentation est celle de RAJA
//! reduce values in block into thread 0
template <typename ReduceOperator,typename T>
ARCCORE_DEVICE inline T block_reduce(T val, T identity)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = getThreadId();

  int warpId = threadId % WARP_SIZE;
  int warpNum = threadId / WARP_SIZE;

  T temp = val;

  if (numThreads % WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < WARP_SIZE; i *= 2) {
      T rhs = impl::shfl_xor_sync(temp, i);
      ReduceOperator::apply(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < WARP_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = impl::shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        ReduceOperator::apply(temp, rhs);
      }
    }
  }

  //printf("CONTINUE tid=%d wid=%d wnum=%d\n",threadId,warpId,warpNum);

  // reduce per warp values
  if (numThreads > WARP_SIZE) {

    __shared__ T sd[MAX_WARPS];

    // write per warp values to shared memory
    if (warpId == 0) {
      sd[warpNum] = temp;
    }

    __syncthreads();

    if (warpNum == 0) {

      // read per warp values
      if (warpId * WARP_SIZE < numThreads) {
        temp = sd[warpId];
      } else {
        temp = identity;
      }
      for (int i = 1; i < WARP_SIZE; i *= 2) {
        T rhs = impl::shfl_xor_sync(temp, i);
        ReduceOperator::apply(temp, rhs);
      }
    }

    __syncthreads();
  }
  return temp;
}

template<typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorSum<DataType>::
_applyDevice(DataType* ptr,DataType v,DataType identity)
{
  DataType rv = impl::block_reduce<impl::SimpleSumReduceOperator<DataType>>(v,identity);
  // Seul le thread 0 de chaque bloc contient la valeur réduite
  // On utilise ensuite une opération atomique pour la réduction
  // entre les blocs.
  // TODO: utiliser une réduction sur la grille via des valeurs temporaires
  // allouées sur le GPU.
  if (impl::getThreadId()==0){
    //printf("Adding value %d\n",rv);
    impl::CommonCudaHipAtomicAdd<DataType>::apply(ptr,rv);
  }
}

template<typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorMax<DataType>::
_applyDevice(DataType* ptr,DataType v,DataType identity)
{
  DataType rv = impl::block_reduce<impl::SimpleMaxReduceOperator<DataType>>(v,identity);
  if (impl::getThreadId()==0){
    impl::CommonCudaHipAtomicMax<DataType>::apply(ptr,rv);
  }
}

template<typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorMin<DataType>::
_applyDevice(DataType* ptr,DataType v,DataType identity)
{
  //_doPrintMin(v,identity);
  DataType rv = impl::block_reduce<impl::SimpleMinReduceOperator<DataType>>(v,identity);
  if (impl::getThreadId()==0){
    //_doPrintMin(rv,identity);
    impl::CommonCudaHipAtomicMin<DataType>::apply(ptr,rv);
    //_doPrintMin(*ptr,identity);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
