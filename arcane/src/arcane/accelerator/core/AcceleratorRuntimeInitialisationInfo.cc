// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.cc                     (C) 2000-2024 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Property.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/DeviceId.h"
#include "arcane/utils/ConcurrencyUtils.h"

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

template<typename V> void AcceleratorRuntimeInitialisationInfo::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();
  p << b.addString("AcceleratorRuntime")
        .addDescription("Name of the accelerator runtime (currently only 'cuda', 'hip' or 'sycl') to use")
        .addCommandLineArgument("AcceleratorRuntime")
        .addGetter([](auto a) { return a.x.acceleratorRuntime(); })
        .addSetter([](auto a) { a.x.setAcceleratorRuntime(a.v); });
}

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
  if (&rhs!=this){
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
  if (a=="cuda")
    return eExecutionPolicy::CUDA;
  if (a=="hip")
    return eExecutionPolicy::HIP;
  if (a=="sycl")
    return eExecutionPolicy::SYCL;
  if (!a.null())
    return eExecutionPolicy::None;
  if (TaskFactory::isActive())
    return eExecutionPolicy::Thread;
  return eExecutionPolicy::Sequential;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneInitializeRunner(Accelerator::Runner& runner,ITraceMng* tm,
                       const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  using namespace Accelerator;
  String accelerator_runtime = acc_info.acceleratorRuntime();
  eExecutionPolicy policy = acc_info.executionPolicy();
  if (policy==eExecutionPolicy::None)
    ARCANE_FATAL("Invalid policy eExecutionPolicy::None");
  tm->info() << "AcceleratorRuntime=" << accelerator_runtime;
  if (impl::isAcceleratorPolicy(policy)){
    tm->info() << "Using accelerator runtime=" << policy << " device=" << acc_info.deviceId();
    runner.initialize(policy,acc_info.deviceId());
    runner.setAsCurrentDevice();
  }
  else{
    tm->info() << "Using accelerator runtime=" << policy;
    runner.initialize(policy);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(AcceleratorRuntimeInitialisationInfo,());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

