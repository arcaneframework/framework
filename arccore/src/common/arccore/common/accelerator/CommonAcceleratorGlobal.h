// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobal.h                                     (C) 2000-2026 */
/*                                                                           */
/* General declarations for accelerator support.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_COMMONACCELERATORCOREGLOBAL_H
#define ARCCORE_COMMON_ACCELERATOR_COMMONACCELERATORCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"
#include "arccore/trace/TraceGlobal.h"

#include <iosfwd>

/*!
 * \file AcceleratorCoreGlobal.h
 *
 * This file contains the declarations of the types for the
 * 'arcane_accelerator_core' component.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAcceleratorMng;
class Runner;
class RunQueue;
class RunQueuePool;
class RunCommand;
class RunQueueEvent;
class AcceleratorRuntimeInitialisationInfo;
class RunQueueBuildInfo;
class MemoryCopyArgs;
class MemoryPrefetchArgs;
class DeviceId;
class DeviceInfo;
class DeviceMemoryInfo;
class ProfileRegion;
class IDeviceInfoList;
class PointerAttribute;
class ViewBuildInfo;
class RunnerInternal;
enum class eMemoryAdvice;

namespace Impl
{
  class RuntimeStaticInfo;
  class IRunnerRuntime;
  // typedef for compatibility with older versions (October 2022)
  using IRunQueueRuntime = IRunnerRuntime;
  class IRunQueueStream;
  class RunCommandImpl;
  class IReduceMemoryImpl;
  class ReduceMemoryImpl;
  class RunQueueImpl;
  class IRunQueueEventImpl;
  class RunnerImpl;
  class RunQueueImplStack;
  class KernelLaunchArgs;
  class RunCommandLaunchInfo;
  class NativeStream;
  class CudaUtils;
  class HipUtils;
  class SyclUtils;
} // namespace Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution policy for a Runner.
 */
enum class eExecutionPolicy
{
  //! No execution policy
  None,
  //! Sequential execution policy
  Sequential,
  //! Multi-threaded execution policy
  Thread,
  //! Execution policy using the CUDA environment
  CUDA,
  //! Execution policy using the HIP environment
  HIP,
  //! Execution policy using the SYCL environment
  SYCL
};

//! Prints the name of the execution policy
extern "C++" ARCCORE_COMMON_EXPORT
std::ostream&
operator<<(std::ostream& o, eExecutionPolicy exec_policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reduction operation policy on accelerators.
 *
 * \note Starting from Arcane version 3.15, only the Grid policy is available.
 */
enum class eDeviceReducePolicy
{
  /*!
   * \brief Uses atomic operations between blocks.
   *
   * \deprecated This policy is no longer available. If specified, it will
   * behave like eDeviceReducePolicy::Grid.
   */
  Atomic ARCCORE_DEPRECATED_REASON("Y2025: Use eDeviceReducePolicy::Grid instead") = 1,
  //! Uses a compute kernel with synchronization between blocks.
  Grid = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Predefined priority levels for run queues
 *        on accelerators
 */
enum class eRunQueuePriority : int
{
  //! Uses 0 as the default value
  Default = 0,
  //! An arbitrary negative value to define a high priority
  High = -100,
  //! An arbitrary positive value to define a low priority
  Low = 100
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Memory type for a pointer
enum class ePointerMemoryType
{
  //NOTE: The values are equivalent to cudaMemoryType. If
  // these values are changed, the corresponding function
  // in the runtime (getPointerAttribute()) must be changed.
  Unregistered = 0,
  Host = 1,
  Device = 2,
  Managed = 3
};

//! Prints the name of the memory type
extern "C++" ARCCORE_COMMON_EXPORT
std::ostream&
operator<<(std::ostream& o, ePointerMemoryType mem_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accessibility information of a memory address.
 *
 * Indicates whether a memory address is accessible on an accelerator or
 * on the CPU.
 *
 * \sa getPointerAccessibility()
 */
enum class ePointerAccessibility
{
  //! Unknown accessibility
  Unknown = 0,
  //! Not accessible
  No = 1,
  //! Accessible
  Yes = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if \a exec_policy corresponds to an accelerator
inline bool
isAcceleratorPolicy(eExecutionPolicy exec_policy)
{
  return exec_policy == eExecutionPolicy::CUDA || exec_policy == eExecutionPolicy::HIP || exec_policy == eExecutionPolicy::SYCL;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accessibility of address \a ptr for execution on queue \a queue.
 *
 * If \a queue is null, returns ePointerAccessibility::Unknown.
 * If \a ptr_attr is not null, it will be filled with pointer information
 * as if Runner::fillPointerAttribute() had been called.
 */
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
getPointerAccessibility(RunQueue* queue, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibility of address \a ptr for execution on \a runner.
 *
 * If \a runner is null, returns ePointerAccessibility::Unknown.
 * If \a ptr_attr is not null, it will be filled with pointer information
 * as if Runner::fillPointerAttribute() had been called.
 */
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
getPointerAccessibility(Runner* runner, const void* ptr, PointerAttribute* ptr_attr = nullptr);

/*!
 * \brief Accessibility of address \a ptr for execution policy\a policy.
 *
 * If \a ptr_attr is not null, it will be filled with pointer information
 * as if Runner::fillPointerAttribute() had been called.
 */
extern "C++" ARCCORE_COMMON_EXPORT ePointerAccessibility
getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr = nullptr);

//! Accessibility of address \a ptr for execution on \a queue_or_runner_or_policy.
template <typename T> inline ePointerAccessibility
getPointerAccessibility(T& queue_or_runner_or_policy, const void* ptr, PointerAttribute* ptr_attr = nullptr)
{
  return getPointerAccessibility(&queue_or_runner_or_policy, ptr, ptr_attr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*!
 * \brief Checks if \a ptr is accessible for execution on \a queue.
 *
 * Raises a FatalErrorException if it is not.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(const RunQueue* queue, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Checks if \a ptr is accessible for execution on \a runner.
 *
 * Raises a FatalErrorException if it is not.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(const Runner* runner, const void* ptr,
                               const char* name, const TraceInfo& ti);

/*!
 * \brief Checks if \a ptr is accessible for execution \a policy.
 *
 * Raises a FatalErrorException if it is not.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arcaneCheckPointerIsAccessible(eExecutionPolicy policy, const void* ptr,
                               const char* name, const TraceInfo& ti);

inline void
arcaneCheckPointerIsAccessible(const RunQueue& queue, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  arcaneCheckPointerIsAccessible(&queue, ptr, name, ti);
}

inline void
arcaneCheckPointerIsAccessible(const Runner& runner, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  arcaneCheckPointerIsAccessible(&runner, ptr, name, ti);
}

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Macro that checks if \a ptr is accessible for a RunQueue or a Runner.
 *
 * Raises an exception if it is not.
 */
#define ARCCORE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue_or_runner_or_policy, ptr) \
  ::Arcane::Accelerator::Impl::arcaneCheckPointerIsAccessible((queue_or_runner_or_policy), (ptr), #ptr, A_FUNCINFO)

#define ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue_or_runner_or_policy, ptr) \
  ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS((queue_or_runner_or_policy), (ptr))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_CHECK)

//! Macro that checks in check mode if \a ptr is accessible for a RunQueue or a Runner.
#define ARCCORE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr) \
  ARCCORE_CHECK_ACCESSIBLE_POINTER_ALWAYS((queue_or_runner_or_policy), (ptr))

#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr) \
  ARCCORE_CHECK_ACCESSIBLE_POINTER((queue_or_runner_or_policy), (ptr))

//! Macro that checks in check mode if \a ptr is accessible for a RunQueue or a Runner.
#else

//! Macro that checks in check mode if \a ptr is accessible for a RunQueue or a Runner.
#define ARCCORE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr)

#define ARCANE_CHECK_ACCESSIBLE_POINTER(queue_or_runner_or_policy, ptr)

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

// For compatibility with older versions of Arcane.
// To be deprecated by the end of 2026.
using Arcane::Accelerator::isAcceleratorPolicy;
} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
