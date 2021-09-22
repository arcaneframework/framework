// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.cc                     (C) 2000-2021 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Property.h"

#include "arcane/accelerator/core/Runner.h"
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename V> void AcceleratorRuntimeInitialisationInfo::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();
  p << b.addString("AcceleratorRuntime")
        .addDescription("Name of the accelerator runtime (currently only 'cuda') to use")
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

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneInitializeRunner(Accelerator::Runner& runner,ITraceMng* tm,
                       const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  using namespace Accelerator;
  String accelerator_runtime = acc_info.acceleratorRuntime();
  tm->info() << "AcceleratorRuntime=" << accelerator_runtime;
  if (accelerator_runtime=="cuda"){
    tm->info() << "Using CUDA runtime";
    runner.setExecutionPolicy(eExecutionPolicy::CUDA);
  }
  else if (TaskFactory::isActive()){
    tm->info() << "Using Task runtime";
    runner.setExecutionPolicy(eExecutionPolicy::Thread);
  }
  else{
    tm->info() << "Using Sequential runtime";
    runner.setExecutionPolicy(eExecutionPolicy::Sequential);
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

