// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLauncher.h                                            (C) 2000-2025 */
/*                                                                           */
/* Gestion du lancement des noyaux de calcul sur accélérateur.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_KERNELLAUNCHER_H
#define ARCCORE_ACCELERATOR_KERNELLAUNCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CheckedConvert.h"
#include "arccore/base/ForLoopRanges.h"

#include "arccore/common/accelerator/NativeStream.h"
#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arccore/accelerator/AcceleratorUtils.h"
#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)
#define ARCCORE_KERNEL_CUDA_FUNC(a) a
#else
#define ARCCORE_KERNEL_CUDA_FUNC(a) Arcane::Accelerator::Impl::invalidKernel
#endif

#if defined(ARCCORE_COMPILING_HIP)
#define ARCCORE_KERNEL_HIP_FUNC(a) a
#else
#define ARCCORE_KERNEL_HIP_FUNC(a) Arcane::Accelerator::Impl::invalidKernel
#endif

#if defined(ARCCORE_COMPILING_SYCL)
#define ARCCORE_KERNEL_SYCL_FUNC(a) a
#else
#define ARCCORE_KERNEL_SYCL_FUNC(a) Arcane::Accelerator::Impl::InvalidKernelClass
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour appliquer une opération pour les arguments supplémentaires
 * en début et en fin de noyau CUDA/HIP.
 */
class CudaHipKernelRemainingArgsHelper
{
 public:

  //! Applique les fonctors des arguments additionnels en début de kernel
  template <typename... RemainingArgs> static inline ARCCORE_DEVICE void
  applyAtBegin(Int32 index, RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(index, remaining_args), ...);
  }

  //! Applique les fonctors des arguments additionnels en fin de kernel
  template <typename... RemainingArgs> static inline ARCCORE_DEVICE void
  applyAtEnd(Int32 index, RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(index, remaining_args), ...);
  }

 private:

  template <typename OneArg> static inline ARCCORE_DEVICE void
  _doOneAtBegin(Int32 index, OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    HandlerType::execWorkItemAtBeginForCudaHip(one_arg, index);
  }
  template <typename OneArg> static inline ARCCORE_DEVICE void
  _doOneAtEnd(Int32 index, OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    HandlerType::execWorkItemAtEndForCudaHip(one_arg, index);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour appliquer une opération pour les arguments supplémentaires
 * en début et en fin de noyau Sycl.
 */
class SyclKernelRemainingArgsHelper
{
 public:

#if defined(ARCCORE_COMPILING_SYCL)
  //! Applique les fonctors des arguments additionnels en début de kernel
  template <typename... RemainingArgs> static inline ARCCORE_HOST_DEVICE void
  applyAtBegin(sycl::nd_item<1> x, SmallSpan<std::byte> shm_view,
               RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(x, shm_view, remaining_args), ...);
  }

  //! Applique les fonctors des arguments additionnels en fin de kernel
  template <typename... RemainingArgs> static inline void
  applyAtEnd(sycl::nd_item<1> x, SmallSpan<std::byte> shm_view,
             RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(x, shm_view, remaining_args), ...);
  }

 private:

  template <typename OneArg> static void
  _doOneAtBegin(sycl::nd_item<1> x, SmallSpan<std::byte> shm_memory, OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    if constexpr (requires { HandlerType::execWorkItemAtBeginForSycl(one_arg, x, shm_memory); })
      HandlerType::execWorkItemAtBeginForSycl(one_arg, x, shm_memory);
    else
      HandlerType::execWorkItemAtBeginForSycl(one_arg, x);
  }
  template <typename OneArg> static void
  _doOneAtEnd(sycl::nd_item<1> x, SmallSpan<std::byte> shm_memory, OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    if constexpr (requires { HandlerType::execWorkItemAtBeginForSycl(one_arg, x, shm_memory); })
      HandlerType::execWorkItemAtEndForSycl(one_arg, x, shm_memory);
    else
      HandlerType::execWorkItemAtEndForSycl(one_arg, x);
  }

#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
struct Privatizer
{
  using value_type = T;
  using reference_type = value_type&;
  value_type m_private_copy;

  ARCCORE_HOST_DEVICE Privatizer(const T& o)
  : m_private_copy{ o }
  {}
  ARCCORE_HOST_DEVICE reference_type privateCopy() { return m_private_copy; }
};

template <typename T>
ARCCORE_HOST_DEVICE auto privatize(const T& item) -> Privatizer<T>
{
  return Privatizer<T>{ item };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename TraitsType, typename Lambda, typename... RemainingArgs> __global__ void
doIndirectGPULambda2(SmallSpan<const Int32> ids, Lambda func, RemainingArgs... remaining_args)
{
  using BuilderType = TraitsType::BuilderType;
  using LocalIdType = BuilderType::ValueType;

  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  if (i < ids.size()) {
    LocalIdType lid(ids[i]);
    body(BuilderType::create(i, lid), remaining_args...);
  }
  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

template <typename ItemType, typename Lambda, typename... RemainingArgs> __global__ void
doDirectGPULambda2(Int32 vsize, Lambda func, RemainingArgs... remaining_args)
{
  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  if (i < vsize) {
    body(i, remaining_args...);
  }
  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCCORE_COMPILING_CUDA_OR_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

//! Boucle 1D avec indirection
template <typename TraitsType, typename Lambda, typename... RemainingArgs>
class DoIndirectSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shared_memory,
                  SmallSpan<const Int32> ids, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    using BuilderType = TraitsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    SyclKernelRemainingArgsHelper::applyAtBegin(x, shared_memory, remaining_args...);
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid), remaining_args...);
    }
    SyclKernelRemainingArgsHelper::applyAtEnd(x, shared_memory, remaining_args...);
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
  ARCCORE_FATAL("Invalid kernel");
}

template<typename Lambda,typename... LambdaArgs>
class InvalidKernelClass
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)
template <typename... KernelArgs> inline void
_applyKernelCUDAVariadic(bool is_cooperative, const KernelLaunchArgs& tbi,
                         cudaStream_t& s, Int32 shared_memory,
                         const void* kernel_ptr, KernelArgs... args)
{
  void* all_args[] = { (reinterpret_cast<void*>(&args))... };
  if (is_cooperative)
    cudaLaunchCooperativeKernel(kernel_ptr, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), all_args, shared_memory, s);
  else
    cudaLaunchKernel(kernel_ptr, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), all_args, shared_memory, s);
}
#endif

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
 */
template <typename CudaKernel, typename Lambda, typename LambdaArgs, typename... RemainingArgs> void
_applyKernelCUDA(RunCommandLaunchInfo& launch_info, const CudaKernel& kernel, Lambda& func,
                 const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args)
{
#if defined(ARCCORE_COMPILING_CUDA)
  Int32 shared_memory = launch_info._sharedMemorySize();
  const void* kernel_ptr = reinterpret_cast<const void*>(kernel);
  auto tbi = launch_info._threadBlockInfo(kernel_ptr, shared_memory);
  cudaStream_t s = CudaUtils::toNativeStream(launch_info._internalNativeStream());
  bool is_cooperative = launch_info._isUseCooperativeLaunch();
  bool use_cuda_launch = launch_info._isUseCudaLaunchKernel();
  if (use_cuda_launch || is_cooperative)
    _applyKernelCUDAVariadic(is_cooperative, tbi, s, shared_memory, kernel_ptr, args, func, other_args...);
  else {
    // TODO: utiliser cudaLaunchKernel() à la place.
    kernel<<<tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), shared_memory, s>>>(args, func, other_args...);
  }
#else
  ARCCORE_UNUSED(launch_info);
  ARCCORE_UNUSED(kernel);
  ARCCORE_UNUSED(func);
  ARCCORE_UNUSED(args);
  ARCCORE_FATAL_NO_CUDA_COMPILATION();
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
_applyKernelHIP(RunCommandLaunchInfo& launch_info, const HipKernel& kernel, const Lambda& func,
                const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... other_args)
{
#if defined(ARCCORE_COMPILING_HIP)
  Int32 wanted_shared_memory = launch_info._sharedMemorySize();
  auto tbi = launch_info._threadBlockInfo(reinterpret_cast<const void*>(kernel), wanted_shared_memory);
  hipStream_t s = HipUtils::toNativeStream(launch_info._internalNativeStream());
  hipLaunchKernelGGL(kernel, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), wanted_shared_memory, s, args, func, other_args...);
#else
  ARCCORE_UNUSED(launch_info);
  ARCCORE_UNUSED(kernel);
  ARCCORE_UNUSED(func);
  ARCCORE_UNUSED(args);
  ARCCORE_FATAL_NO_HIP_COMPILATION();
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
void _applyKernelSYCL(RunCommandLaunchInfo& launch_info, SyclKernel kernel, Lambda& func,
                      const LambdaArgs& args, [[maybe_unused]] const RemainingArgs&... remaining_args)
{
#if defined(ARCCORE_COMPILING_SYCL)
  sycl::queue s = SyclUtils::toNativeStream(launch_info._internalNativeStream());
  sycl::event event;
  if constexpr (IsAlwaysUseSyclNdItem<LambdaArgs>::value || sizeof...(RemainingArgs) > 0) {
    auto tbi = launch_info.kernelLaunchArgs();
    Int32 b = tbi.nbBlockPerGrid();
    Int32 t = tbi.nbThreadPerBlock();
    sycl::nd_range<1> loop_size(b * t, t);
    Int32 wanted_shared_memory = launch_info._sharedMemorySize();
    // TODO: regarder s'il y a un coût à utiliser à chaque fois
    // 'sycl::local_accessor' même si on n'a pas besoin de mémoire partagée.
    event = s.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<std::byte> shm_acc(sycl::range<1>(wanted_shared_memory), cgh);
      cgh.parallel_for(loop_size, [=](sycl::nd_item<1> i) {
        std::byte* shm_ptr = shm_acc.get_multi_ptr<sycl::access::decorated::no>().get();
        kernel(i, SmallSpan<std::byte>(shm_ptr, wanted_shared_memory), args, func, remaining_args...);
      });
    });
    //event = s.parallel_for(loop_size, [=](sycl::nd_item<1> i) { kernel(i, args, func, remaining_args...); });
  }
  else {
    sycl::range<1> loop_size = launch_info.totalLoopSize();
    event = s.parallel_for(loop_size, [=](sycl::id<1> i) { kernel(i, args, func); });
  }
  launch_info._addSyclEvent(&event);
#else
  ARCCORE_UNUSED(launch_info);
  ARCCORE_UNUSED(kernel);
  ARCCORE_UNUSED(func);
  ARCCORE_UNUSED(args);
  ARCCORE_FATAL_NO_SYCL_COMPILATION();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_MACRO_PARENS ()

// Les trois macros suivantes permettent de générer récursivement un ensemble
// de paramètres. Si on veut supporter plus de paramètres, on peut ajouter
// des appels à la macro suivante dans chaque macro.
// Plus d'info ici: https://stackoverflow.com/questions/70238923/how-to-expand-a-recursive-macro-via-va-opt-in-a-nested-context
#define ARCCORE_MACRO_EXPAND(...) ARCCORE_MACRO_EXPAND2(ARCCORE_MACRO_EXPAND2(ARCCORE_MACRO_EXPAND2(__VA_ARGS__)))
#define ARCCORE_MACRO_EXPAND2(...) ARCCORE_MACRO_EXPAND1(ARCCORE_MACRO_EXPAND1(ARCCORE_MACRO_EXPAND1(__VA_ARGS__)))
#define ARCCORE_MACRO_EXPAND1(...) __VA_ARGS__

#define ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER(a1, ...) \
  , decltype(a1)& a1                                                     \
  __VA_OPT__(ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH_AGAIN ARCCORE_MACRO_PARENS(__VA_ARGS__))

#define ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH_AGAIN() ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER

/*
 * \brief Macro pour générer les arguments de la lambda.
 *
 * Cette macro est interne à Arcane et ne doit pas être utilisée en dehors de Arcane.
 *
 * Cette macro permet de générer pour chaque argument \a arg une valeur `decltype(arg)& arg`.
 *
 * Par exemple:
 * \code
 * ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH(value1,value2)
 * // Cela génère le code suivant:
 * , decltype(value1)&, decltype(value2)&
 * \encode
 */
#define ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH(...) \
  __VA_OPT__(ARCCORE_MACRO_EXPAND(ARCCORE_RUNCOMMAND_REDUCER_FOR_EACH_HELPER(__VA_ARGS__)))


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
