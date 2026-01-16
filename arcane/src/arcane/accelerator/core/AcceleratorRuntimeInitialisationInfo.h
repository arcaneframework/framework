// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.h                      (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_ACCELERATORRUNTIMEINITIALISATIONINFO_H
#define ARCANE_ACCELERATOR_CORE_ACCELERATORRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"
#include "arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialise \a runner avec les informations de \a acc_info.
 *
 * Cette fonction appelle Accelerator::Runner::setAsCurrentDevice() après
 * l'initialisation.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneInitializeRunner(Runner& runner, ITraceMng* tm,
                       const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

