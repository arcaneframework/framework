// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueInternal.h                                          (C) 2000-2022 */
/*                                                                           */
/* Implémentation de la gestion d'une file d'exécution sur accélérateur.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNQUEUEINTERNAL_H
#define ARCANE_ACCELERATOR_RUNQUEUEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/LoopRanges.h"

#include "arcane/accelerator/core/IRunnerRuntime.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

#if defined(ARCANE_COMPILING_HIP)
#include <hip/hip_runtime.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
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

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

template<typename ItemType,typename Lambda> __global__
void doIndirectGPULambda(SmallSpan<const Int32> ids,Lambda func)
{
  typedef typename ItemType::LocalIdType LocalIdType;

  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<ids.size()){
    LocalIdType lid(ids[i]);
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(lid);
  }
}

template<typename ItemType,typename Lambda> __global__
void doDirectGPULambda(Int32 vsize,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<vsize){
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(i);
  }
}

template<typename LoopBoundType,typename Lambda> __global__
void doDirectGPULambdaArrayBounds(LoopBoundType bounds,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<bounds.nbElement()){
    body(bounds.getIndices(i));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_COMPILING_CUDA || ARCANE_COMPILING_HIP

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

#if defined(ARCANE_COMPILING_CUDA)
/*!
 * \brief Fonction générique pour exécuter un kernel CUDA.
 *
 * \param kernel noyau CUDA
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 */
template<typename CudaKernel,typename Lambda,typename LambdaArgs> void
_applyKernelCUDA(impl::RunCommandLaunchInfo& launch_info,const CudaKernel& kernel, Lambda& func,const LambdaArgs& args)
{
  launch_info.beginExecute();
  auto [b,t] = launch_info.threadBlockInfo();
  cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
  // TODO: utiliser cudaLaunchKernel() à la place.
  kernel <<<b, t, 0, *s>>>(args,func);
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)
/*!
 * \brief Fonction générique pour exécuter un kernel CUDA.
 *
 * \param kernel noyau CUDA
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 */
template<typename HipKernel,typename Lambda,typename LambdaArgs> void
_applyKernelHIP(impl::RunCommandLaunchInfo& launch_info,const HipKernel& kernel, Lambda& func,const LambdaArgs& args)
{
  launch_info.beginExecute();
  auto [b,t] = launch_info.threadBlockInfo();
  hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
  hipLaunchKernelGGL(kernel, b, t, 0, *s, args, func);
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
