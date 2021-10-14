// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyCompactPolicy.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Politique de compactage des entités.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/ItemFamilyCompactInfos.h"

#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyCompactPolicy::
ItemFamilyCompactPolicy(ItemFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
beginCompact(ItemFamilyCompactInfos& compact_infos)
{
  m_family->beginCompactItems(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos)
{
  m_family->compactVariablesAndGroups(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
endCompact(ItemFamilyCompactInfos& compact_infos)
{
  m_family->finishCompactItems(compact_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyCompactPolicy::
compactConnectivityData()
{
  m_family->compactReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemFamilyCompactPolicy::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardItemFamilyCompactPolicy::
StandardItemFamilyCompactPolicy(ItemFamily* family)
: ItemFamilyCompactPolicy(family)
{
  IMesh* mesh = family->mesh();

  m_node_family = mesh->nodeFamily();
  m_edge_family = mesh->edgeFamily();
  m_face_family = mesh->faceFamily();
  m_cell_family = mesh->cellFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardItemFamilyCompactPolicy::
updateInternalReferences(IMeshCompacter*)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
