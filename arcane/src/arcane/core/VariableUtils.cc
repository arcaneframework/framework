// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtils.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires diverses sur les variables.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableUtils.h"

#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/IMemoryAllocator.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/core/IData.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/VariableRef.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

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
prefetchVariableAsync(IVariable* var, const RunQueue* queue_or_null)
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
  // On ne pré-charge que si la variable est en mémoire unifiée.
  // Cela n'est utile que dans ce cas et de plus avec CUDA cela provoque
  // une erreur si la mémoire n'est pas u
  if (nd->memoryAllocator()->memoryResource() != eMemoryResource::UnifiedMemory)
    return;
  ConstMemoryView mem_view = nd->memoryView();
  queue_or_null->prefetchMemory(MemoryPrefetchArgs(mem_view).addAsync());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
prefetchVariableAsync(VariableRef& var, const RunQueue* queue_or_null)
{
  return prefetchVariableAsync(var.variable(), queue_or_null);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
markVariableAsMostlyReadOnly(IVariable* var)
{
  DataAllocationInfo alloc_info(eMemoryLocationHint::HostAndDeviceMostlyRead);
  var->setAllocationInfo(alloc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
markVariableAsMostlyReadOnly(VariableRef& var)
{
  return markVariableAsMostlyReadOnly(var.variable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
markVariableAsMostlyReadOnly(::Arcane::Materials::MeshMaterialVariableRef& var)
{
  auto vars = var.materialVariable()->_internalApi()->variableReferenceList();
  for (VariableRef* v : vars)
    markVariableAsMostlyReadOnly(v->variable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
experimentalChangeAllocator(::Arcane::Materials::IMeshMaterialVariable* var,
                            eMemoryRessource mem)
{
  MemoryAllocationOptions mem_opts(MemoryUtils::getAllocationOptions(mem));
  Arcane::Materials::IMeshMaterialVariableInternal* mat_var = var->_internalApi();
  for (VariableRef* vref : mat_var->variableReferenceList())
    vref->variable()->_internalApi()->changeAllocator(mem_opts);
  var->globalVariable()->_internalApi()->changeAllocator(mem_opts);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
experimentalChangeAllocator(IVariable* var, eMemoryRessource mem)
{
  MemoryAllocationOptions mem_opts(MemoryUtils::getAllocationOptions(mem));
  var->_internalApi()->changeAllocator(mem_opts);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtils::
experimentalChangeAllocator(VariableRef& var, eMemoryRessource mem)
{
  experimentalChangeAllocator(var.variable(), mem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
