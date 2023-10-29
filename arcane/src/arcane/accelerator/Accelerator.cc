// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Accelerator.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/accelerator/Reduce.h"

#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/accelerator/SpanViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file RunCommandLoop.h
 *
 * \brief Types et macros pour gérer les boucles sur les accélérateurs
 */

/*!
 * \file RunCommandEnumerate.h
 *
 * \brief Types et macros pour gérer les énumérations des entités sur les accélérateurs
 */

/*!
 * \file RunCommandMaterialEnumerate.h
 *
 * \brief Types et macros pour gérer les énumérations des matériaux et
 * milieux sur les accélérateurs
 */

/*!
 * \file Reduce.h
 *
 * \brief Types et fonctions pour gérer les synchronisations sur les accélérateurs
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::Accelerator::
initializeRunner(Runner& runner,ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info)
{
  arcaneInitializeRunner(runner,tm,acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::String Arcane::Accelerator::impl::
getBadPolicyMessage(eExecutionPolicy policy)
{
  switch(policy){
  case eExecutionPolicy::CUDA:
    return "Requesting CUDA kernel execution but the kernel is not compiled with CUDA."
    " You need to compile the file containing this kernel with CUDA compiler.";
  case eExecutionPolicy::HIP:
    return "Requesting HIP kernel execution but the kernel is not compiled with HIP."
    " You need to compile the file containing this kernel with HIP compiler.";
  default:
    break;
  }
  return String::format("Invalid execution policy '{0}'", policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
