// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemListBuilder.cc                                 (C) 2000-2015 */
/*                                                                           */
/* Classe d'aide à la construction d'une liste de ComponentItem.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/ComponentItemListBuilder.h"
#include "arcane/materials/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemListBuilder::
ComponentItemListBuilder(MeshMaterialVariableIndexer* var_indexer,
                         Integer begin_index_in_partial)
: m_component_index(var_indexer->index()+1)
, m_index_in_partial(begin_index_in_partial)
, m_indexer(var_indexer)
{
  Integer reserve_size = 4000;
  m_pure_matvar_indexes.reserve(reserve_size);
  m_partial_matvar_indexes.reserve(reserve_size);
  m_partial_local_ids.reserve(reserve_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
