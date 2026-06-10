// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLauncher.h                                            (C) 2000-2026 */
/*                                                                           */
/* Management of kernel launch on accelerator.                               */
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

// The following macros are used to apply a kernel if the associated backend
// is available. Otherwise, an exception is raised.
// These macros are used, for example, in RunCommandLoop.

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
 * \brief Class to apply an operation for additional arguments
 * at the beginning and end of a CUDA/HIP kernel.
 */
class CudaHipKernelRemainingArgsHelper
{
 public:

  //! Applies the functors of additional arguments at the beginning of the kernel
  template <typename... RemainingArgs> static inline ARCCORE_DEVICE void
  applyAtBegin(Int32 index, RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(index, remaining_args), ...);
  }

  //! Applies the functors of additional arguments at the end of the kernel
  template <typename... RemainingArgs> static inline ARCCORE_DEVICE void
  applyAtEnd(Int32 index, RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(index, remaining_args), ...);
  }

  //! Indicates if one of the additional arguments requires a barrier.
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
 * \brief Class to apply an operation for additional arguments
 * at the beginning and end of a Sycl kernel.
 */
class SyclKernelRemainingArgsHelper
{
 public:

#if defined(ARCCORE_COMPILING_SYCL)
  //! Applies the functors of additional arguments at the beginning of the kernel
  template <typename... RemainingArgs> static inline ARCCORE_HOST_DEVICE void
  applyAtBegin(sycl::nd_item<1> x, SmallSpan<std::byte> shm_view,
               RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(x, shm_view, remaining_args), ...);
  }

  //! Applies the functors of additional arguments at the end of the kernel
  template <typename... RemainingArgs> static inline void
  applyAtEnd(sycl::nd_item<1> x, SmallSpan<std::byte> shm_view,
             RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(x, shm_view, remaining_args), ...);
  }

  //! Indicates if one of the additional arguments requires a barrier.
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

// More used
// Empty function to simulate an invalid kernel because it was not compiled with
// the adequate compiler. Should normally not be called.
template <typename Lambda, typename... LambdaArgs>
inline void invalidKernel(Lambda&, const LambdaArgs&...)
{
  ARCCORE_FATAL("Invalid kernel");
}
// More used
template <typename Lambda, typename... LambdaArgs>
class InvalidKernelClass
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA)

//! Static class for launching CUDA kernels.
class CudaKernelLauncher
{
 public:

  /*!
   * \brief Generic function to execute a CUDA kernel.
   *
   * \param kernel CUDA kernel
   * \param func function to be executed by the kernel
   * \param bounds iteration range
   * \param other_args other lambda arguments
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
      _applyKernelCUDAVariadic(is_cooperative, tbi, s, kernel_ptr, bounds, func, other_args...);
    else {
      kernel<<<tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), shared_memory, s>>>(bounds, func, other_args...);
    }
  }

 private:

  template <typename... KernelArgs> static inline void
  _applyKernelCUDAVariadic(bool is_cooperative, const KernelLaunchArgs& tbi,
                           cudaStream_t& s, const void* kernel_ptr, KernelArgs... args)
  {
    void* all_args[] = { (reinterpret_cast<void*>(&args))... };
    if (is_cooperative)
      cudaLaunchCooperativeKernel(kernel_ptr, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), all_args, tbi.sharedMemorySize(), s);
    else
      cudaLaunchKernel(kernel_ptr, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), all_args, tbi.sharedMemorySize(), s);
  }
};

#endif // ARCCORE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_HIP)

//! Static class for launching ROCM/HIP kernels
class HipKernelLauncher
{
 public:

  /*!
   * \brief Generic function to execute a HIP kernel.
   *
   * \param kernel HIP kernel
   * \param func function to be executed by the kernel
   * \param bounds iteration range
   * \param other_args other lambda arguments
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
    bool is_cooperative = launch_info._isUseCooperativeLaunch();
    if (is_cooperative) {
      _applyCooperativeKernel(tbi, s, kernel_ptr, bounds, func, other_args...);
    }
    else
      hipLaunchKernelGGL(kernel, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), wanted_shared_memory, s, bounds, func, other_args...);
  }

 private:

  template <typename... KernelArgs> static inline void
  _applyCooperativeKernel(const KernelLaunchArgs& tbi, hipStream_t& s,
                          const void* kernel_ptr, KernelArgs... args)
  {
    void* all_args[] = { (reinterpret_cast<void*>(&args))... };
    // TODO: Check and test the return code
    (void)hipLaunchCooperativeKernel(kernel_ptr, tbi.nbBlockPerGrid(), tbi.nbThreadPerBlock(), all_args, tbi.sharedMemorySize(), s);
  }
};

#endif // ARCCORE_COMPILING_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

//! Static class for launching SYCL kernels
class SyclKernelLauncher
{
 public:

  /*!
   * \brief Generic function to execute a SYCL kernel.
   *
   * \param kernel SYCL kernel
   * \param func function to be executed by the kernel
   * \param bounds iteration range
   * \param other_args other lambda arguments
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
      //TODO: look into how to convert \a kernel into a functor
      auto tbi = launch_info._computeKernelLaunchArgs(nullptr);
      Int32 b = tbi.nbBlockPerGrid();
      Int32 t = tbi.nbThreadPerBlock();
      Int32 wanted_shared_memory = tbi.sharedMemorySize();
      sycl::nd_range<1> loop_size(b * t, t);
      // TODO: check if there is a cost to use 'sycl::local_accessor' every time
      // even if shared memory is not needed.
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

// The following three macros allow recursively generating a set
// of parameters. If you want to support more parameters, you can add
// calls to the following macro in each macro.
// More info here: https://stackoverflow.com/questions/70238923/how-to-expand-a-recursive-macro-via-va-opt-in-a-nested-context
#define ARCCORE_MACRO_EXPAND(...) ARCCORE_MACRO_EXPAND2(ARCCORE_MACRO_EXPAND2(ARCCORE_MACRO_EXPAND2(__VA_ARGS__)))
#define ARCCORE_MACRO_EXPAND2(...) ARCCORE_MACRO_EXPAND1(ARCCORE_MACRO_EXPAND1(ARCCORE_MACRO_EXPAND1(__VA_ARGS__)))
#define ARCCORE_MACRO_EXPAND1(...) __VA_ARGS__

#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER(a1, ...) \
  , decltype(a1)& a1 __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_AGAIN ARCCORE_MACRO_PARENS(__VA_ARGS__))

#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_AGAIN() ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER

/*
 * \brief Macro to generate lambda arguments.
 *
 * This macro is internal to Arcane and should not be used outside of Arcane.
 *
 * This macro allows generating a `decltype(arg)& arg` value for each argument \a arg.
 *
 * For example:
 * \code
 * ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(value1,value2)
 * // This generates the following code:
 * , decltype(value1)&, decltype(value2)&
 * \encode
 */
#define ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(...) \
  __VA_OPT__(ARCCORE_MACRO_EXPAND(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH_HELPER(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
