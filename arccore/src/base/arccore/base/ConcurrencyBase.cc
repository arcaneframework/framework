// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyBase.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Classes de base pour la gestion du multi-threading.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ConcurrencyBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ConcurrencyBase::m_max_allowed_thread = 1;
ParallelLoopOptions ConcurrencyBase::m_default_loop_options;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le nombre maximum de thread à utiliser.
 *
 * Cette méthode doit être appelée par l'implémentation de ITaskImplementation
 * lors de l'initialisation. Il ne faut plus la modifier ensuite.
 */
void ConcurrencyBase::
_setMaxAllowedThread(Int32 v)
{
  if (v < 0)
    v = 1;
  m_max_allowed_thread = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
