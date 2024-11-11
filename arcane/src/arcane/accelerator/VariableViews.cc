﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableViews.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Gestion des vues sur les variables pour les accélérateurs.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/VariableViews.h"

#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"

#include "arcane/core/VariableUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file VariableViews.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs des variables du maillage.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableViewBase::
VariableViewBase(const ViewBuildInfo& vbi, IVariable* var)
{
  impl::RunQueueImpl* q = vbi._internalQueue();
  if (q->isAutoPrefetchCommand()) {
    RunQueue rq(q);
    VariableUtils::prefetchVariableAsync(var, &rq);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
