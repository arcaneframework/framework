// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLauncher.h                                            (C) 2000-2024 */
/*                                                                           */
/* Gestion du lancement des noyaux de calcul sur accélérateur.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_KERNELLAUNCHER_H
#define ARCANE_ACCELERATOR_KERNELLAUNCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/LoopRanges.h"

#include "arcane/accelerator/core/NativeStream.h"
#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"
#include "arcane/accelerator/AcceleratorUtils.h"

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

#if defined(ARCANE_COMPILING_SYCL)
#define ARCANE_KERNEL_SYCL_FUNC(a) a
#else
#define ARCANE_KERNEL_SYCL_FUNC(a) Arcane::Accelerator::impl::InvalidKernelClass
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
/*!
 * \brief Classe pour appliquer la finalisation pour les arguments supplémentaires.
 */
class KernelRemainingArgsHelper
{
 public:

  //! Applique les fonctors des arguments additionnels.
  template <typename... RemainingArgs> static inline ARCCORE_DEVICE void
  applyRemainingArgs(Int32 index, RemainingArgs&... remaining_args)
  {
    // Applique les réductions
    (remaining_args._internalExecWorkItem(index), ...);
  }

#if defined(ARCANE_COMPILING_SYCL)
  //! Applique les fonctors des arguments additionnels.
  template <typename... RemainingArgs> static inline ARCCORE_HOST_DEVICE void
  applyRemainingArgs(sycl::nd_item<1> x, RemainingArgs&... remaining_args)
  {
    // Applique les réductions
    (remaining_args._internalExecWorkItem(x), ...);
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

template <typename BuilderType, typename Lambda> __global__ void
doIndirectGPULambda(SmallSpan<const Int32> ids, Lambda func)
{
  using LocalIdType = BuilderType::ValueType;

  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < ids.size()) {
    LocalIdType lid(ids[i]);
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(BuilderType::create(i, lid));
  }
}

template <typename ItemType, typename Lambda> __global__ void
doDirectGPULambda(Int32 vsize, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vsize) {
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(i);
  }
}

template <typename LoopBoundType, typename Lambda> __global__ void
doDirectGPULambdaArrayBounds(LoopBoundType bounds, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < bounds.nbElement()) {
    body(bounds.getIndices(i));
  }
}

template <typename TraitsType, typename Lambda, typename... RemainingArgs> __global__ void
doIndirectGPULambda2(SmallSpan<const Int32> ids, Lambda func, RemainingArgs... remaining_args)
{
  using BuilderType = TraitsType::BuilderType;
  using LocalIdType = BuilderType::ValueType;

  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < ids.size()) {
    LocalIdType lid(ids[i]);
    body(BuilderType::create(i, lid), remaining_args...);
  }
  KernelRemainingArgsHelper::applyRemainingArgs(i, remaining_args...);
}

template <typename ItemType, typename Lambda, typename... RemainingArgs> __global__ void
doDirectGPULambda2(Int32 vsize, Lambda func, RemainingArgs... remaining_args)
{
  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vsize) {
    body(i, remaining_args...);
  }
  KernelRemainingArgsHelper::applyRemainingArgs(i, remaining_args...);
}

template <typename LoopBoundType, typename Lambda, typename... RemainingArgs> __global__ void
doDirectGPULambdaArrayBounds2(LoopBoundType bounds, Lambda func, RemainingArgs... remaining_args)
{
  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < bounds.nbElement()) {
    body(bounds.getIndices(i), remaining_args...);
  }
  KernelRemainingArgsHelper::applyRemainingArgs(i, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_COMPILING_CUDA || ARCANE_COMPILING_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

//! Boucle N-dimension sans indirection
template <typename LoopBoundType, typename Lambda, typename... RemainingArgs>
class DoDirectSYCLLambdaArrayBounds
{
 public:

  void operator()(sycl::nd_item<1> x, LoopBoundType bounds, Lambda func, RemainingArgs... remaining_args) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    if (i < bounds.nbElement()) {
      body(bounds.getIndices(i), remaining_args...);
    }
    KernelRemainingArgsHelper::applyRemainingArgs(x, remaining_args...);
  }
  void operator()(sycl::id<1> x, LoopBoundType bounds, Lambda func) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x);
    if (i < bounds.nbElement()) {
      body(bounds.getIndices(i));
    }
  }
};

//! Boucle 1D avec indirection
template <typename TraitsType, typename Lambda, typename... RemainingArgs>
class DoIndirectSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<const Int32> ids, Lambda func, RemainingArgs... remaining_args) const
  {
    using BuilderType = TraitsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid), remaining_args...);
    }
    KernelRemainingArgsHelper::applyRemainingArgs(x, remaining_args...);
  }
  void operator()(sycl::id<1> x, SmallSpan<const Int32> ids, Lambda func) const
  {
    using BuilderType = TraitsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x);
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid));
    }
  }
};

#endif

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

template<typename Lambda,typename... LambdaArgs>
class InvalidKernelClass
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fonction générique pour exécuter un kernel CUDA.
 *
 * \param kernel noyau CUDA
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 * 
 * TODO: Tester si Lambda est bien une fonction, le SFINAE étant peu lisible :
 * typename std::enable_if_t<std::is_function_v<std::decay_t<Lambda> > >* = nullptr
 * attendons les concepts c++20 (requires)
 * 
 */
template <typename CudaKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs> void
_applyKernelCUDA(impl::RunCommandLaunchInfo& launch_info, const CudaKernel& kernel, Lambda& func,
                 const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args)
{
#if defined(ARCANE_COMPILING_CUDA)
  Int32 wanted_shared_memory = 0;
  auto tbi = launch_info._threadBlockInfo(reinterpret_cast<const void*>(kernel), wanted_shared_memory);
  cudaStream_t s = CudaUtils::toNativeStream(launch_info._internalNativeStream());
  // TODO: utiliser cudaLaunchKernel() à la place.
  kernel<<<tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), wanted_shared_memory, s>>>(args, func, other_args...);
#else
  ARCANE_UNUSED(launch_info);
  ARCANE_UNUSED(kernel);
  ARCANE_UNUSED(func);
  ARCANE_UNUSED(args);
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
template <typename HipKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs> void
_applyKernelHIP(impl::RunCommandLaunchInfo& launch_info, const HipKernel& kernel, const Lambda& func,
                const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args)
{
#if defined(ARCANE_COMPILING_HIP)
  Int32 wanted_shared_memory = 0;
  auto tbi = launch_info._threadBlockInfo(reinterpret_cast<const void*>(kernel), wanted_shared_memory);
  hipStream_t s = HipUtils::toNativeStream(launch_info._internalNativeStream());
  hipLaunchKernelGGL(kernel, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), wanted_shared_memory, s, args, func, other_args...);
#else
  ARCANE_UNUSED(launch_info);
  ARCANE_UNUSED(kernel);
  ARCANE_UNUSED(func);
  ARCANE_UNUSED(args);
  ARCANE_FATAL_NO_HIP_COMPILATION();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction générique pour exécuter un kernel SYCL.
 *
 * \param kernel noyau SYCL
 * \param func fonction à exécuter par le noyau
 * \param args arguments de la fonction lambda
 */
template <typename SyclKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs>
void _applyKernelSYCL(impl::RunCommandLaunchInfo& launch_info, SyclKernel kernel, Lambda& func,
                      const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... remaining_args)
{
#if defined(ARCANE_COMPILING_SYCL)
  sycl::queue s = SyclUtils::toNativeStream(launch_info._internalNativeStream());
  sycl::event event;
  if constexpr (sizeof...(RemainingArgs) > 0) {
    auto tbi = launch_info.kernelLaunchArgs();
    Int32 b = tbi.nbBlockPerGrid();
    Int32 t = tbi.nbThreadPerBlock();
    sycl::nd_range<1> loop_size(b * t, t);
    event = s.parallel_for(loop_size, [=](sycl::nd_item<1> i) { kernel(i, args, func, remaining_args...); });
  }
  else {
    sycl::range<1> loop_size = launch_info.totalLoopSize();
    event = s.parallel_for(loop_size, [=](sycl::id<1> i) { kernel(i, args, func); });
  }
  launch_info._addSyclEvent(&event);
#else
  ARCANE_UNUSED(launch_info);
  ARCANE_UNUSED(kernel);
  ARCANE_UNUSED(func);
  ARCANE_UNUSED(args);
  ARCANE_FATAL_NO_SYCL_COMPILATION();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_MACRO_PARENS ()

// Les trois macros suivantes permettent de générer récursivement un ensemble
// de paramètres. Si on veut supporter plus de paramètres, on peut ajouter
// des appels à la macro suivante dans chaque macro.
// Plus d'info ici: https://stackoverflow.com/questions/70238923/how-to-expand-a-recursive-macro-via-va-opt-in-a-nested-context
#define ARCANE_MACRO_EXPAND(...) ARCANE_MACRO_EXPAND2(ARCANE_MACRO_EXPAND2(ARCANE_MACRO_EXPAND2(__VA_ARGS__)))
#define ARCANE_MACRO_EXPAND2(...) ARCANE_MACRO_EXPAND1(ARCANE_MACRO_EXPAND1(ARCANE_MACRO_EXPAND1(__VA_ARGS__)))
#define ARCANE_MACRO_EXPAND1(...) __VA_ARGS__

#define ARCANE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER(a1, ...) \
  , decltype(a1)& a1                                                     \
  __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH_AGAIN ARCANE_MACRO_PARENS(__VA_ARGS__))

#define ARCANE_RUNCOMMAND_REDUCER_FOR_EACH_AGAIN() ARCANE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER

/*
 * \brief Macro pour générer les arguments de la lambda.
 *
 * Cette macro est interne à Arcane et ne doit pas être utilisée en dehors de Arcane.
 *
 * Cette macro permet de générer pour chaque argument \a arg une valeur `decltype(arg)& arg`.
 *
 * Par exemple:
 * \code
 * ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(value1,value2)
 * // Cela génère le code suivant:
 * , decltype(value1)&, decltype(value2)&
 * \encode
 */
#define ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(...) \
  __VA_OPT__(ARCANE_MACRO_EXPAND(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER(__VA_ARGS__)))


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
