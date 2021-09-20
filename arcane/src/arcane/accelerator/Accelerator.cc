// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Accelerator.cc                                              (C) 2000-2021 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/accelerator/core/Runner.h"

#include "arcane/accelerator/Reduce.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/Concurrency.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_EXPORT void Arcane::Accelerator::
initializeRunner(Runner& runner,ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info)
{
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
