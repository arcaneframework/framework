// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

#include "arcane/accelerator/Reduce.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_EXPORT void Arcane::Accelerator::
initializeRunner(Runner& runner,ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  arcaneInitializeRunner(runner,tm,acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
