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

#include "arcane/accelerator/core/internal/AcceleratorRuntimeInitialisationInfoProperties.h"
#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"

#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename V> void AcceleratorRuntimeInitialisationInfoProperties::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();
  p << b.addString("AcceleratorRuntime")
        .addDescription("Name of the accelerator runtime (currently only 'cuda', 'hip' or 'sycl') to use")
        .addCommandLineArgument("AcceleratorRuntime")
        .addGetter([](auto a) { return a.x.acceleratorRuntime(); })
        .addSetter([](auto a) { a.x.setAcceleratorRuntime(a.v); });
  p << b.addBool("UseAccelerator")
       .addDescription("activate/deactivate accelerator runtime")
       .addCommandLineArgument("UseAccelerator")
       .addGetter([](auto a) { return a.x.isUsingAcceleratorRuntime(); })
       .addSetter([](auto a) { a.x.setIsUsingAcceleratorRuntime(a.v); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneInitializeRunner(Accelerator::Runner& runner,ITraceMng* tm,
                       const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  Impl::arccoreInitializeRunner(runner,tm,acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(AcceleratorRuntimeInitialisationInfoProperties, ());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

