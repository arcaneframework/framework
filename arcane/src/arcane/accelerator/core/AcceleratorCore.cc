// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCore.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"

#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"

#include "arcane/accelerator/core/DeviceInfoList.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/ViewBuildInfo.h"
#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/RunQueue.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \namespace Arcane::Accelerator
 *
 * \brief Espace de nom pour l'utilisation des accélérateurs.
 *
 * Toutes les classes et types utilisés pour la gestion des accélérateurs
 * sont dans ce namespace.
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
  impl::IRunnerRuntime* global_cuda_runqueue_runtime = nullptr;
  bool global_is_using_hip_runtime = false;
  impl::IRunnerRuntime* global_hip_runqueue_runtime = nullptr;
  bool global_is_using_sycl_runtime = false;
  impl::IRunnerRuntime* global_sycl_runqueue_runtime = nullptr;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool impl::
isUsingCUDARuntime()
{
  return global_is_using_cuda_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setUsingCUDARuntime(bool v)
{
  global_is_using_cuda_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation CUDA de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT impl::IRunnerRuntime* impl::
getCUDARunQueueRuntime()
{
  return global_cuda_runqueue_runtime;
}

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setCUDARunQueueRuntime(IRunnerRuntime* v)
{
  global_cuda_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool impl::
isUsingHIPRuntime()
{
  return global_is_using_hip_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setUsingHIPRuntime(bool v)
{
  global_is_using_hip_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation HIP de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT impl::IRunnerRuntime* impl::
getHIPRunQueueRuntime()
{
  return global_hip_runqueue_runtime;
}

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setHIPRunQueueRuntime(impl::IRunnerRuntime* v)
{
  global_hip_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool impl::
isUsingSYCLRuntime()
{
  return global_is_using_sycl_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setUsingSYCLRuntime(bool v)
{
  global_is_using_sycl_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT impl::IRunnerRuntime* impl::
getSYCLRunQueueRuntime()
{
  return global_hip_runqueue_runtime;
}

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void impl::
setSYCLRunQueueRuntime(impl::IRunnerRuntime* v)
{
  global_hip_runqueue_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Affiche le nom de la politique d'exécution
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
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

extern "C++" impl::IRunnerRuntime* impl::
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

ePointerAccessibility impl::RuntimeStaticInfo::
getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr)
{
  // Regarde si le pointeur est accessible pour la politique d'exécution donnée.
  // Le seul cas où on peut le savoir exactement est si on a un runtime
  // accélérateur et que la valeur retournée par getPointeAttribute() est valide.
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

void impl::RuntimeStaticInfo::
checkPointerIsAcccessible(eExecutionPolicy policy, const void* ptr,
                          const char* name, const TraceInfo& ti)
{
  // Le pointeur nul est toujours accessible.
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
  return impl::RuntimeStaticInfo::getPointerAccessibility(policy, ptr, ptr_attr);
}

void impl::
arcaneCheckPointerIsAccessible(eExecutionPolicy policy, const void* ptr,
                               const char* name, const TraceInfo& ti)
{
  return impl::RuntimeStaticInfo::checkPointerIsAcccessible(policy, ptr, name, ti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::
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
