// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLauncher.h                                            (C) 2000-2026 */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Les macros suivantes servent à appliquer un noyau si le backend associé
// est disponible. Sinon, on lève une exception.
// Ces macros sont utilisées par exemple dans RunCommandLoop.

#if defined(ARCCORE_COMPILING_CUDA)
#define ARCCORE_KERNEL_CUDA_FUNC(kernel, ...) ::Arcane::Accelerator::Impl::CudaKernelLauncher::apply(kernel, __VA_ARGS__)
#else
#define ARCCORE_KERNEL_CUDA_FUNC(kernel, ...) ARCCORE_FATAL_NO_CUDA_COMPILATION()
#endif

#if defined(ARCCORE_COMPILING_HIP)
#define ARCCORE_KERNEL_HIP_FUNC(kernel, ...) ::Arcane::Accelerator::Impl::HipKernelLauncher::apply(kernel, __VA_ARGS__)
#else
#define ARCCORE_KERNEL_HIP_FUNC(kernel, ...) ARCCORE_FATAL_NO_HIP_COMPILATION()
#endif

#if defined(ARCCORE_COMPILING_SYCL)
#define ARCCORE_KERNEL_SYCL_FUNC(kernel, ...) ::Arcane::Accelerator::Impl::SyclKernelLauncher::apply(kernel, __VA_ARGS__)
#else
#define ARCCORE_KERNEL_SYCL_FUNC(kernel, ...) ARCCORE_FATAL_NO_SYCL_COMPILATION()
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

  //! Indique si un des arguments supplémentaires nécessite une barrière.
  template <typename... RemainingArgs> static inline bool
  isNeedBarrier(const RemainingArgs&... remaining_args)
  {
    bool is_need_barrier = (_isOneNeedBarrier(remaining_args) || ...);
    return is_need_barrier;
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
  template <typename OneArg> static inline bool
  _isOneNeedBarrier([[maybe_unused]] const OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    return HandlerType::isNeedBarrier(one_arg);
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

  //! Indique si un des arguments supplémentaires nécessite une barrière.
  template <typename... RemainingArgs> static inline bool
  isNeedBarrier(const RemainingArgs&... remaining_args)
  {
    bool is_need_barrier = (_isOneNeedBarrier(remaining_args) || ...);
    return is_need_barrier;
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
  template <typename OneArg> static inline bool
  _isOneNeedBarrier([[maybe_unused]] const OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    return HandlerType::isNeedBarrier(one_arg);
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

template <typename Lambda>
void doDirectThreadLambda(Integer begin, Integer size, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  for (Int32 i = 0; i < size; ++i) {
    func(begin + i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Plus utilisé
// Fonction vide pour simuler un noyau invalide car non compilé avec
// le compilateur adéquant. Ne devrait normalement pas être appelé.
template <typename Lambda, typename... LambdaArgs>
inline void invalidKernel(Lambda&, const LambdaArgs&...)
{
  ARCCORE_FATAL("Invalid kernel");
}
// Plus utilisé
template <typename Lambda, typename... LambdaArgs>
class InvalidKernelClass
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)

//! Classe statique pour lancer les kernels CUDA.
class CudaKernelLauncher
{
 public:

  /*!
   * \brief Fonction générique pour exécuter un kernel CUDA.
   *
   * \param kernel noyau CUDA
   * \param func fonction à exécuter par le noyau
   * \param bounds intervalle d'itération
   * \param other_args autres arguments de la lambda
   */
  template <typename CudaKernel, typename Lambda, typename LoopBoundType, typename... RemainingArgs> static void
  apply(const CudaKernel& kernel, RunCommandLaunchInfo& launch_info, Lambda& func,
        const LoopBoundType& bounds, const RemainingArgs&... other_args)
  {
    const void* kernel_ptr = reinterpret_cast<const void*>(kernel);
    auto tbi = launch_info._computeKernelLaunchArgs(kernel_ptr);
    Int32 shared_memory = tbi.sharedMemorySize();
    cudaStream_t s = CudaUtils::toNativeStream(launch_info._internalNativeStream());
    bool is_cooperative = launch_info._isUseCooperativeLaunch();
    bool use_cuda_launch = launch_info._isUseCudaLaunchKernel();
    bool is_need_barrier = CudaHipKernelRemainingArgsHelper::isNeedBarrier(other_args...);
    launch_info._setIsNeedBarrier(is_need_barrier);
    if (use_cuda_launch || is_cooperative)
      _applyKernelCUDAVariadic(is_cooperative, tbi, s, shared_memory, kernel_ptr, bounds, func, other_args...);
    else {
      kernel<<<tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), shared_memory, s>>>(bounds, func, other_args...);
    }
  }

 private:

  template <typename... KernelArgs> static inline void
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
};

#endif // ARCCORE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_HIP)

//! Classe statique pour lancer les kernels ROCM/HIP
class HipKernelLauncher
{
 public:

  /*!
   * \brief Fonction générique pour exécuter un kernel HIP.
   *
   * \param kernel noyau HIP
   * \param func fonction à exécuter par le noyau
   * \param bounds intervalle d'itération
   * \param other_args autres arguments de la lambda
   */
  template <typename HipKernel, typename Lambda, typename LoopBoundType, typename... RemainingArgs> static void
  apply(const HipKernel& kernel, RunCommandLaunchInfo& launch_info, const Lambda& func,
        const LoopBoundType& bounds, const RemainingArgs&... other_args)
  {
    const void* kernel_ptr = reinterpret_cast<const void*>(kernel);
    auto tbi = launch_info._computeKernelLaunchArgs(kernel_ptr);
    Int32 wanted_shared_memory = tbi.sharedMemorySize();
    hipStream_t s = HipUtils::toNativeStream(launch_info._internalNativeStream());
    bool is_need_barrier = CudaHipKernelRemainingArgsHelper::isNeedBarrier(other_args...);
    launch_info._setIsNeedBarrier(is_need_barrier);
    hipLaunchKernelGGL(kernel, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), wanted_shared_memory, s, bounds, func, other_args...);
  }
};

#endif // ARCCORE_COMPILING_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

//! Classe statique pour lancer les kernels SYCL
class SyclKernelLauncher
{
 public:

  /*!
   * \brief Fonction générique pour exécuter un kernel SYCL.
   *
   * \param kernel noyau SYCL
   * \param func fonction à exécuter par le noyau
   * \param bounds intervalle d'itération
   * \param other_args autres arguments de la lambda
   */
  template <typename SyclKernel, typename Lambda, typename LoopBoundType, typename... RemainingArgs> static void
  apply(SyclKernel kernel, RunCommandLaunchInfo& launch_info, Lambda& func,
        const LoopBoundType& bounds, const RemainingArgs&... remaining_args)
  {
    sycl::queue s = SyclUtils::toNativeStream(launch_info._internalNativeStream());
    sycl::event event;
    bool is_need_barrier = SyclKernelRemainingArgsHelper::isNeedBarrier(remaining_args...);
    launch_info._setIsNeedBarrier(is_need_barrier);
    if constexpr (IsAlwaysUseSyclNdItem<LoopBoundType>::value || sizeof...(RemainingArgs) > 0) {
      //TODO: regarder comment convertir \a kernel en un functor
      auto tbi = launch_info._computeKernelLaunchArgs(nullptr);
      Int32 b = tbi.nbBlockPerGrid();
      Int32 t = tbi.nbThreadPerBlock();
      Int32 wanted_shared_memory = tbi.sharedMemorySize();
      sycl::nd_range<1> loop_size(b * t, t);
      // TODO: regarder s'il y a un coût à utiliser à chaque fois
      // 'sycl::local_accessor' même si on n'a pas besoin de mémoire partagée.
      event = s.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<std::byte> shm_acc(sycl::range<1>(wanted_shared_memory), cgh);
        cgh.parallel_for(loop_size, [=](sycl::nd_item<1> i) {
          std::byte* shm_ptr = shm_acc.get_multi_ptr<sycl::access::decorated::no>().get();
          kernel(i, SmallSpan<std::byte>(shm_ptr, wanted_shared_memory), bounds, func, remaining_args...);
        });
      });
      //event = s.parallel_for(loop_size, [=](sycl::nd_item<1> i) { kernel(i, args, func, remaining_args...); });
    }
    else {
      sycl::range<1> loop_size = launch_info.totalLoopSize();
      event = s.parallel_for(loop_size, [=](sycl::id<1> i) { kernel(i, bounds, func); });
    }
    launch_info._addSyclEvent(&event);
  }
};

#endif // ARCCORE_COMPILING_SYCL

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

#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER(a1, ...) \
  , decltype(a1)& a1 __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_AGAIN ARCCORE_MACRO_PARENS(__VA_ARGS__))

#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_AGAIN() ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER

/*
 * \brief Macro pour générer les arguments de la lambda.
 *
 * Cette macro est interne à Arcane et ne doit pas être utilisée en dehors de Arcane.
 *
 * Cette macro permet de générer pour chaque argument \a arg une valeur `decltype(arg)& arg`.
 *
 * Par exemple:
 * \code
 * ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(value1,value2)
 * // Cela génère le code suivant:
 * , decltype(value1)&, decltype(value2)&
 * \encode
 */
#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(...) \
  __VA_OPT__(ARCCORE_MACRO_EXPAND(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
