// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemListBuilder.cc                                 (C) 2000-2024 */
/*                                                                           */
/* Classe d'aide à la construction d'une liste de ComponentItem.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

#include "arcane/utils/MemoryUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
namespace
{
  MemoryAllocationOptions _allocInfo(const String& name)
  {
    MemoryAllocationOptions opts(MemoryUtils::getDefaultDataAllocator());
    opts.setArrayName(name);
    return opts;
  }
} // namespace
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemListBuilder::
ComponentItemListBuilder()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemListBuilderOld::
ComponentItemListBuilderOld(MeshMaterialVariableIndexer* var_indexer,
                         Integer begin_index_in_partial)
: m_component_index(var_indexer->index() + 1)
, m_index_in_partial(begin_index_in_partial)
, m_pure_matvar_indexes(_allocInfo("PureMatVarIndexes"))
, m_partial_matvar_indexes(_allocInfo("PartialMatVarIndexes"))
, m_partial_local_ids(_allocInfo("PartialLocalIds"))
, m_indexer(var_indexer)
{
  Integer reserve_size = 4000;
  m_pure_matvar_indexes.reserve(reserve_size);
  m_partial_matvar_indexes.reserve(reserve_size);
  m_partial_local_ids.reserve(reserve_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
