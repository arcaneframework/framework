// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyBase.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Base classes for multi-threading management.                              */
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
 * \brief Sets the maximum number of threads to use.
 *
 * This method must be called by the ITaskImplementation implementation
 * during initialization. It should not be modified afterward.
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
