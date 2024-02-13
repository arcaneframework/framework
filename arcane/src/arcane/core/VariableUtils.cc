// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtils.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires diverses sur les variables.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableUtils.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/core/IData.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/VariableRef.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file VariableUtils.h
 *
 * \brief Fonctions utilitaires sur les variables.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
prefetchVariableAsync(IVariable* var, RunQueue* queue_or_null)
{
  ARCANE_CHECK_POINTER(var);
  if (!queue_or_null)
    return;
  if (!var->isUsed())
    return;
  IData* d = var->data();
  INumericDataInternal* nd = d->_commonInternal()->numericData();
  if (!nd)
    return;
  ConstMemoryView mem_view = nd->memoryView();
  queue_or_null->prefetchMemory(MemoryPrefetchArgs(mem_view).addAsync());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
prefetchVariableAsync(VariableRef& var, RunQueue* queue_or_null)
{
  return prefetchVariableAsync(var.variable(), queue_or_null);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
