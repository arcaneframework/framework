// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.cc                     (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h"

#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ConcurrencyBase.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/common/MemoryUtils.h"
#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/DeviceId.h"
#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AcceleratorRuntimeInitialisationInfo::Impl
{
 public:

  bool m_is_using_accelerator_runtime = false;
  String m_accelerator_runtime;
  DeviceId m_device_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo::
AcceleratorRuntimeInitialisationInfo()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo::
AcceleratorRuntimeInitialisationInfo(const AcceleratorRuntimeInitialisationInfo& rhs)
: m_p(new Impl(*rhs.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo& AcceleratorRuntimeInitialisationInfo::
operator=(const AcceleratorRuntimeInitialisationInfo& rhs)
{
  if (&rhs != this) {
    delete m_p;
    m_p = new Impl(*(rhs.m_p));
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo::
~AcceleratorRuntimeInitialisationInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AcceleratorRuntimeInitialisationInfo::
isUsingAcceleratorRuntime() const
{
  return m_p->m_is_using_accelerator_runtime;
}

void AcceleratorRuntimeInitialisationInfo::
setIsUsingAcceleratorRuntime(bool v)
{
  m_p->m_is_using_accelerator_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String AcceleratorRuntimeInitialisationInfo::
acceleratorRuntime() const
{
  return m_p->m_accelerator_runtime;
}

void AcceleratorRuntimeInitialisationInfo::
setAcceleratorRuntime(StringView v)
{
  m_p->m_accelerator_runtime = v;
  if (!v.empty())
    setIsUsingAcceleratorRuntime(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DeviceId AcceleratorRuntimeInitialisationInfo::
deviceId() const
{
  return m_p->m_device_id;
}

void AcceleratorRuntimeInitialisationInfo::
setDeviceId(DeviceId v)
{
  m_p->m_device_id = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eExecutionPolicy AcceleratorRuntimeInitialisationInfo::
executionPolicy() const
{
  String a = acceleratorRuntime();
  if (a == "cuda")
    return eExecutionPolicy::CUDA;
  if (a == "hip")
    return eExecutionPolicy::HIP;
  if (a == "sycl")
    return eExecutionPolicy::SYCL;
  if (!a.null())
    return eExecutionPolicy::None;
  if (ConcurrencyBase::maxAllowedThread() > 1)
    return eExecutionPolicy::Thread;
  return eExecutionPolicy::Sequential;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::
arccoreInitializeRunner(Accelerator::Runner& runner, ITraceMng* tm,
                        const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  using namespace Accelerator;
  String accelerator_runtime = acc_info.acceleratorRuntime();
  eExecutionPolicy policy = acc_info.executionPolicy();
  if (policy == eExecutionPolicy::None)
    ARCCORE_FATAL("Invalid policy eExecutionPolicy::None");
  tm->info() << "AcceleratorRuntime=" << accelerator_runtime;
  tm->info() << "DefaultDataAllocator MemoryResource=" << MemoryUtils::getDefaultDataMemoryResource();
  if (impl::isAcceleratorPolicy(policy)) {
    tm->info() << "Using accelerator runtime=" << policy << " device=" << acc_info.deviceId();
    runner.initialize(policy, acc_info.deviceId());
    runner.setAsCurrentDevice();
  }
  else {
    tm->info() << "Using accelerator runtime=" << policy;
    runner.initialize(policy);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
