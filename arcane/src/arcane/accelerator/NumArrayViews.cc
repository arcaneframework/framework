// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Gestion des vues sur les 'NumArray' pour les accélérateurs.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/NumArrayViews.h"

#include "arcane/utils/MemoryView.h"

#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Memory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file NumArrayViews.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs de la classe 'NumArray'.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayViewBase::
NumArrayViewBase(const ViewBuildInfo& vbi, Span<const std::byte> bytes)
{
  const RunQueue& q = vbi.queue();
  if (q._isAutoPrefetchCommand()) {
    ConstMemoryView mem_view(bytes);
    q.prefetchMemory(MemoryPrefetchArgs(mem_view).addAsync());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
