// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueInternal.h                                          (C) 2000-2024 */
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
#include "arcane/accelerator/RunCommandLaunchInfo.h"

#if defined(ARCANE_COMPILING_HIP)
#include <hip/hip_runtime.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA)
#define ARCANE_KERNEL_CUDA_FUNC(a) a
#else
#define ARCANE_KERNEL_CUDA_FUNC(a) Arcane::Accelerator::impl::invalidKernel
#endif

#if defined(ARCANE_COMPILING_HIP)
#define ARCANE_KERNEL_HIP_FUNC(a) a
#else
#define ARCANE_KERNEL_HIP_FUNC(a) Arcane::Accelerator::impl::invalidKernel
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

template<typename BuilderType,typename Lambda> __global__
void doIndirectGPULambda(SmallSpan<const Int32> ids,Lambda func)
{
  using LocalIdType = BuilderType::ValueType;

  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<ids.size()){
    LocalIdType lid(ids[i]);
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(BuilderType::create(i,lid));
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

// Fonction vide pour simuler un noyau invalide car non compilé avec
// le compilateur adéquant. Ne devrait normalement pas être appelé.
template<typename Lambda,typename... LambdaArgs>
inline void invalidKernel(Lambda&,const LambdaArgs&...)
{
  ARCANE_FATAL("Invalid kernel");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fonction générique pour exécuter un kernel CUDA.
 *
 * \param kernel noyau CUDA
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 * 
 * TODO: Tester si Lambda est bien un fonction, le SFINAE étant peu lisible :
 * typename std::enable_if_t<std::is_function_v<std::decay_t<Lambda> > >* = nullptr
 * attendons les concepts c++20 (requires)
 * 
 */
template<typename CudaKernel,typename Lambda,typename... LambdaArgs> void
_applyKernelCUDA(impl::RunCommandLaunchInfo& launch_info,const CudaKernel& kernel,Lambda& func,[[maybe_unused]] const LambdaArgs&... args)
{
#if defined(ARCANE_COMPILING_CUDA)
  auto [b,t] = launch_info.threadBlockInfo();
  cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
  // TODO: utiliser cudaLaunchKernel() à la place.
  /*
   * Test si dessous non concluant. Le principe de la construction de l'array est bon,
   * l'ensemble ressemble à ce qu'on peut trouver (par exemple :
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/gpu_kernel_helper.h)
   * mais je pense que la lambda pose problème...
   * A creuser encore donc...
   *
    std::array<void*, sizeof...(args)+1> kernel_args{std::forward<void*>((void*)&args)...};
    kernel_args[sizeof...(args)] = std::forward<void*>((void*)&func);
    cudaLaunchKernel<CudaKernel>(kernel, b, t, kernel_args.data(), 0, *s);
   */
  kernel <<<b, t, 0, *s>>>(args...,func);
#else
  ARCANE_UNUSED(launch_info);
  ARCANE_UNUSED(kernel);
  ARCANE_UNUSED(func);
  // ARCANE_UNUSED(args...);  FIXME: ne fonctionne pas, d'où le [[maybe_unused]] dans le prototype
  ARCANE_FATAL_NO_CUDA_COMPILATION();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fonction générique pour exécuter un kernel HIP.
 *
 * \param kernel noyau HIP
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 */
template<typename HipKernel,typename Lambda,typename... LambdaArgs> void
_applyKernelHIP(impl::RunCommandLaunchInfo& launch_info,const HipKernel& kernel,Lambda& func,[[maybe_unused]] const LambdaArgs&... args)
{
#if defined(ARCANE_COMPILING_HIP)
  auto [b,t] = launch_info.threadBlockInfo();
  hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
  hipLaunchKernelGGL(kernel, b, t, 0, *s, args..., func); // TODO: pas encore testé !!!
#else
  ARCANE_UNUSED(launch_info);
  ARCANE_UNUSED(kernel);
  ARCANE_UNUSED(func);
  // ARCANE_UNUSED(args...);  FIXME: ne fonctionne pas, d'où le [[maybe_unused]] dans le prototype
  ARCANE_FATAL_NO_HIP_COMPILATION();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
