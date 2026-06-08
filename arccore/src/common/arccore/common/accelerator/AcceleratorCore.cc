// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCore.cc                                          (C) 2000-2025 */
/*                                                                           */
/* General declarations for accelerator support.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"

#include "arccore/common/accelerator/DeviceInfoList.h"
#include "arccore/common/accelerator/PointerAttribute.h"
#include "arccore/common/accelerator/ViewBuildInfo.h"
#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/common/accelerator/RunQueue.h"

// Not used but necessary for symbol exports.
#include "arccore/common/accelerator/IAcceleratorMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \namespace Arcane::Accelerator
 *
 * \brief Namespace for accelerator usage.
 *
 * All classes and types used for accelerator management
 * are in this namespace.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  bool global_is_using_cuda_runtime = false;
  Impl::IRunnerRuntime* global_cuda_runqueue_runtime = nullptr;
  bool global_is_using_hip_runtime = false;
  Impl::IRunnerRuntime* global_hip_runqueue_runtime = nullptr;
  bool global_is_using_sycl_runtime = false;
  Impl::IRunnerRuntime* global_sycl_runqueue_runtime = nullptr;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT bool Impl::
isUsingCUDARuntime()
{
  return global_is_using_cuda_runtime;
}

extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setUsingCUDARuntime(bool v)
{
  global_is_using_cuda_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Retrieves the CUDA implementation of RunQueue
extern "C++" ARCCORE_COMMON_EXPORT Impl::IRunnerRuntime* Impl::
getCUDARunQueueRuntime()
{
  return global_cuda_runqueue_runtime;
}

//! Sets the CUDA implementation of RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setCUDARunQueueRuntime(IRunnerRuntime* v)
{
  global_cuda_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT bool Impl::
isUsingHIPRuntime()
{
  return global_is_using_hip_runtime;
}

extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setUsingHIPRuntime(bool v)
{
  global_is_using_hip_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Retrieves the HIP implementation of RunQueue
extern "C++" ARCCORE_COMMON_EXPORT Impl::IRunnerRuntime* Impl::
getHIPRunQueueRuntime()
{
  return global_hip_runqueue_runtime;
}

//! Sets the HIP implementation of RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setHIPRunQueueRuntime(Impl::IRunnerRuntime* v)
{
  global_hip_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT bool Impl::
isUsingSYCLRuntime()
{
  return global_is_using_sycl_runtime;
}

extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setUsingSYCLRuntime(bool v)
{
  global_is_using_sycl_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT Impl::IRunnerRuntime* Impl::
getSYCLRunQueueRuntime()
{
  return global_hip_runqueue_runtime;
}

extern "C++" ARCCORE_COMMON_EXPORT void Impl::
setSYCLRunQueueRuntime(Impl::IRunnerRuntime* v)
{
  global_hip_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Displays the name of the execution policy
extern "C++" ARCCORE_COMMON_EXPORT
std::ostream&
operator<<(std::ostream& o, eExecutionPolicy exec_policy)
{
  switch (exec_policy) {
  case eExecutionPolicy::None:
    o << "None";
    break;
  case eExecutionPolicy::Sequential:
    o << "Sequential";
    break;
  case eExecutionPolicy::Thread:
    o << "Thread";
    break;
  case eExecutionPolicy::CUDA:
    o << "CUDA";
    break;
  case eExecutionPolicy::HIP:
    o << "HIP";
    break;
  case eExecutionPolicy::SYCL:
    o << "SYCL";
    break;
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const DeviceId& device_id)
{
  o << device_id.asInt32();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o, ePointerMemoryType mem_type)
{
  switch (mem_type) {
  case ePointerMemoryType::Unregistered:
    o << "Unregistered";
    break;
  case ePointerMemoryType::Host:
    o << "Host";
    break;
  case ePointerMemoryType::Device:
    o << "Device";
    break;
  case ePointerMemoryType::Managed:
    o << "Managed";
    break;
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Impl::IRunnerRuntime* Impl::
getAcceleratorRunnerRuntime()
{
  if (isUsingCUDARuntime())
    return getCUDARunQueueRuntime();
  if (isUsingHIPRuntime())
    return getHIPRunQueueRuntime();
  if (isUsingSYCLRuntime())
    return getSYCLRunQueueRuntime();
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ePointerAccessibility Impl::RuntimeStaticInfo::
getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr)
{
  // Checks if the pointer is accessible for the given execution policy.
  // The only case where we can know exactly is if we have an
  // accelerator runtime and the value returned by getPointeAttribute() is valid.
  if (policy == eExecutionPolicy::None)
    return ePointerAccessibility::Unknown;
  IRunnerRuntime* r = getAcceleratorRunnerRuntime();
  if (!r)
    return ePointerAccessibility::Unknown;
  PointerAttribute attr;
  r->getPointerAttribute(attr, ptr);
  if (ptr_attr) {
    *ptr_attr = attr;
  }
  if (attr.isValid()) {
    if (isAcceleratorPolicy(policy))
      return attr.devicePointer() ? ePointerAccessibility::Yes : ePointerAccessibility::No;
    else {
      if (attr.memoryType() == ePointerMemoryType::Unregistered)
        return ePointerAccessibility::Yes;
      return attr.hostPointer() ? ePointerAccessibility::Yes : ePointerAccessibility::No;
    }
  }
  return ePointerAccessibility::Unknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::RuntimeStaticInfo::
checkPointerIsAcccessible(eExecutionPolicy policy, const void* ptr,
                          const char* name, const TraceInfo& ti)
{
  // The null pointer is always accessible.
  if (!ptr)
    return;
  PointerAttribute ptr_attr;
  ePointerAccessibility a = getPointerAccessibility(policy, ptr, &ptr_attr);
  if (a == ePointerAccessibility::No) {
    auto s = String::format("Pointer 'addr={0}' ({1}) is not accessible "
                            "for this execution policy ({2}).\n  PointerInfo: {3}",
                            ptr, name, policy, ptr_attr);

    throw FatalErrorException(ti, s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ePointerAccessibility
getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr)
{
  return Impl::RuntimeStaticInfo::getPointerAccessibility(policy, ptr, ptr_attr);
}

void Impl::
arcaneCheckPointerIsAccessible(eExecutionPolicy policy, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  return Impl::RuntimeStaticInfo::checkPointerIsAcccessible(policy, ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::
printUUID(std::ostream& o, char bytes[16])
{
  static const char hexa_chars[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

  for (int i = 0; i < 16; ++i) {
    o << hexa_chars[(bytes[i] >> 4) & 0xf];
    o << hexa_chars[bytes[i] & 0xf];
    if (i == 4 || i == 6 || i == 8 || i == 10)
      o << '-';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o, const PointerAttribute& a)
{
  o << "(mem_type=" << a.memoryType() << ", ptr=" << a.originalPointer()
    << ", host_ptr=" << a.hostPointer()
    << ", device_ptr=" << a.devicePointer() << " device=" << a.device() << ")";
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ViewBuildInfo::
ViewBuildInfo(const RunQueue& queue)
: m_queue_impl(queue._internalImpl())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ViewBuildInfo::
ViewBuildInfo(const RunQueue* queue)
: m_queue_impl(queue->_internalImpl())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ViewBuildInfo::
ViewBuildInfo(RunCommand& command)
: m_queue_impl(command._internalQueueImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
