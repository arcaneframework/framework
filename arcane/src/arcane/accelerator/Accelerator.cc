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
