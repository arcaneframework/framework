// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueInternal.h                                          (C) 2000-2021 */
/*                                                                           */
/* Implémentation de la gestion d'une file d'exécution sur accélérateur.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNQUEUEINTERNAL_H
#define ARCANE_ACCELERATOR_RUNQUEUEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/LoopRanges.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/IRunQueueRuntime.h"
#include "arcane/accelerator/NumArray.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

#include "arcane/Concurrency.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: fusionner les parties communes des fonctions applyItems(),
// applyLoop() et applyGenericLoop()

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

template <typename T>
struct Privatizer
{
  using value_type = T;
  using reference_type = value_type&;
  value_type m_private_copy;

  ARCCORE_HOST_DEVICE Privatizer(const T& o) : m_private_copy{o} {}
  ARCCORE_HOST_DEVICE reference_type privateCopy() { return m_private_copy; }
};

template <typename T>
ARCCORE_HOST_DEVICE auto privatize(const T& item) -> Privatizer<T>
{
  return Privatizer<T>{item};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)

template<typename ItemType,typename Lambda> __global__
void doIndirectCUDALambda(Span<const Int32> ids,Lambda func)
{
  typedef typename ItemType::LocalIdType LocalIdType;

  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<ids.size()){
    LocalIdType lid(ids[i]);
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(lid);
  }
}

template<typename ItemType,typename Lambda> __global__
void doDirectCUDALambda(Int64 vsize,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<vsize){
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(i);
  }
}

template<typename LoopBoundType,typename Lambda> __global__
void doDirectCUDALambdaArrayBounds(LoopBoundType bounds,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<bounds.nbElement()){
    func(bounds.getIndices(i));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Lambda>
void doDirectThreadLambda(Integer begin,Integer size,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  for( Int32 i=0; i<size; ++i ){
    func(begin+i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<template<int T> class LoopBoundType,typename Lambda> inline void
applyGenericLoopSequential(LoopBoundType<1> bounds,const Lambda& func)
{
  for( Int64 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    func(ArrayBoundsIndex<1>(i0));
}

template<template<int T> class LoopBoundType,typename Lambda> inline void
applyGenericLoopSequential(LoopBoundType<2> bounds,const Lambda& func)
{
  for( Int64 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int64 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      func(ArrayBoundsIndex<2>(i0,i1));
}

template<template<int T> class LoopBoundType,typename Lambda> inline void
applyGenericLoopSequential(LoopBoundType<3> bounds,const Lambda& func)
{
  for( Int64 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int64 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      for( Int64 i2 = bounds.lowerBound(2); i2 < bounds.upperBound(2); ++i2 )
        func(ArrayBoundsIndex<3>(i0,i1,i2));
}

template<template<int> class LoopBoundType,typename Lambda> inline void
applyGenericLoopSequential(LoopBoundType<4> bounds,const Lambda& func)
{
  for( Int64 i0 = bounds.lowerBound(0); i0 < bounds.upperBound(0); ++i0 )
    for( Int64 i1 = bounds.lowerBound(1); i1 < bounds.upperBound(1); ++i1 )
      for( Int64 i2 = bounds.lowerBound(2); i2 < bounds.upperBound(2); ++i2 )
        for( Int64 i3 = bounds.lowerBound(3); i3 < bounds.upperBound(3); ++i3 )
          func(ArrayBoundsIndex<4>(i0,i1,i2,i3));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
