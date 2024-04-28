﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonCudaHipReduceImpl.h                                   (C) 2000-2024 */
/*                                                                           */
/* Implémentation CUDA et HIP des réductions.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_COMMONCUDHIPAREDUCEIMPL_H
#define ARCANE_ACCELERATOR_COMMONCUDHIPAREDUCEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Ce fichier doit être inclus uniquement par 'arcane/accelerator/Reduce.h'
// et n'est valide que compilé par le compilateur CUDA et HIP

#include "arcane/accelerator/CommonCudaHipAtomicImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention: avec ROCm et un GPU sur bus PCI express la plupart des
// méthodes atomiques ne fonctionnent pas si le pointeur est allouée
// en mémoire unifiée. A priori le problème se pose avec atomicMin, atomicMax,
// atomicInc. Par contre atomicAdd a l'air de fonctionner.

namespace Arcane::Accelerator::impl
{

__device__ __forceinline__ unsigned int getThreadId()
{
  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;
  return threadId;
}

__device__ __forceinline__ unsigned int getBlockId()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  return blockId;
}

#if defined(__HIP__)
constexpr const Int32 WARP_SIZE = warpSize;
#else
constexpr const Int32 WARP_SIZE = 32;
#endif
constexpr const Int32 MAX_BLOCK_SIZE = 1024;
constexpr const Int32 MAX_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;

template <typename T, enum eAtomicOperation>
class SimpleReduceOperator;

template <typename T>
class SimpleReduceOperator<T, eAtomicOperation::Add>
{
 public:

  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = val + v;
  }
};

template <typename T>
class SimpleReduceOperator<T, eAtomicOperation::Max>
{
 public:

  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = v > val ? v : val;
  }
};

template <typename T>
class SimpleReduceOperator<T, eAtomicOperation::Min>
{
 public:

  static ARCCORE_DEVICE inline void apply(T& val, const T v)
  {
    val = v < val ? v : val;
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

ARCCORE_DEVICE inline Int64 shfl_xor_sync(Int64 var, int laneMask)
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

ARCCORE_DEVICE inline Int64 shfl_sync(Int64 var, int laneMask)
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

ARCCORE_DEVICE inline Int64 shfl_xor_sync(Int64 var, int laneMask)
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

ARCCORE_DEVICE inline Int64 shfl_sync(Int64 var, int laneMask)
{
  return ::__shfl(var, laneMask);
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template <typename ReduceOperator, typename T>
ARCCORE_DEVICE inline bool
grid_reduce(T& val,T identity,SmallSpan<T> device_mem,unsigned int* device_count)
{
  int numBlocks = gridDim.x * gridDim.y * gridDim.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  int wrap_around = numBlocks - 1;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  T temp = block_reduce<ReduceOperator>(val, identity);

  // one thread per block writes to device_mem
  bool lastBlock = false;
  if (threadId == 0) {
    device_mem[blockId] = temp;
    // ensure write visible to all threadblocks
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    // Attention: avec ROCm et un GPU sur bus PCI express si 'device_count'
    // est alloué en mémoire unifiée le atomicAdd ne fonctionne pas.
    // Dans ce cas on peut le remplacer par:
    //   unsigned int old_count = ::atomicAdd(device_count, 1) % (wrap_around+1);
    unsigned int old_count = ::atomicInc(device_count, wrap_around);
    lastBlock = ((int)old_count == wrap_around);
  }

  // returns non-zero value if any thread passes in a non-zero value
  lastBlock = __syncthreads_or(lastBlock);

  // last block accumulates values from device_mem
  if (lastBlock) {
    temp = identity;

    for (int i = threadId; i < numBlocks; i += numThreads) {
      ReduceOperator::apply(temp, device_mem[i]);
    }

    temp = block_reduce<ReduceOperator>(temp, identity);

    // one thread returns value
    if (threadId == 0) {
      val = temp;
    }
  }

  return lastBlock && threadId == 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,typename ReduceOperator,typename AtomicReduceOperator>
ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void _applyDeviceGeneric(const ReduceDeviceInfo<DataType>& dev_info)
{
  SmallSpan<DataType> grid_buffer = dev_info.m_grid_buffer;
  DataType identity = dev_info.m_identity;
  unsigned int* device_count = dev_info.m_device_count;
  DataType* ptr = dev_info.m_device_final_ptr;
  DataType v = dev_info.m_current_value;
  bool do_grid_reduce = dev_info.m_use_grid_reduce;

  //if (impl::getThreadId()==0){
  //  printf("BLOCK ID=%d %p s=%d ptr=%p %p use_grid_reduce=%d\n",
  //         getBlockId(),grid_buffer.data(),grid_buffer.size(),ptr,
  //         (void*)device_count,(do_grid_reduce)?1:0);
  //}
  if (do_grid_reduce){
    bool is_done = grid_reduce<ReduceOperator>(v,identity,grid_buffer,device_count);
    if (is_done){
      *ptr = v;
      // Il est important de remettre cette à zéro pour la prochaine utilisation d'un Reducer.
      (*device_count) = 0;
    }
  }
  else{
    DataType rv = impl::block_reduce<ReduceOperator>(v,identity);
    if (impl::getThreadId()==0){
      AtomicReduceOperator::apply(ptr,rv);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorSum<DataType>::
_applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
{
  using ReduceOperator = impl::SimpleReduceOperator<DataType, eAtomicOperation::Add>;
  using AtomicReduceOperator = impl::CommonCudaHipAtomic<DataType, eAtomicOperation::Add>;
  _applyDeviceGeneric<DataType, ReduceOperator, AtomicReduceOperator>(dev_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorMax<DataType>::
_applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
{
  using ReduceOperator = impl::SimpleReduceOperator<DataType,eAtomicOperation::Max>;
  using AtomicReduceOperator = impl::CommonCudaHipAtomic<DataType,eAtomicOperation::Max>;
  _applyDeviceGeneric<DataType,ReduceOperator,AtomicReduceOperator>(dev_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> ARCANE_INLINE_REDUCE ARCCORE_DEVICE
void ReduceFunctorMin<DataType>::
_applyDevice(const ReduceDeviceInfo<DataType>& dev_info)
{
  using ReduceOperator = impl::SimpleReduceOperator<DataType, eAtomicOperation::Min>;
  using AtomicReduceOperator = impl::CommonCudaHipAtomic<DataType, eAtomicOperation::Min>;
  _applyDeviceGeneric<DataType, ReduceOperator, AtomicReduceOperator>(dev_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
