// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityInfo.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Informations sur la connectivité par type d'entité.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemConnectivityInfo.h"
#include "arcane/mesh/ItemSharedInfoList.h"

#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemConnectivityInfo::
ItemConnectivityInfo()
{
  for( Integer i=0; i<NB_ICI; ++i )
    m_infos[i] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityInfo::
fill(ItemSharedInfoList* item_shared_infos,ItemInternalConnectivityList* clist)
{
  m_infos[ICI_Node] = item_shared_infos->maxNodePerItem();
  m_infos[ICI_Edge] = item_shared_infos->maxEdgePerItem();
  m_infos[ICI_Face] = item_shared_infos->maxFacePerItem();
  m_infos[ICI_Cell] = item_shared_infos->maxCellPerItem();

  m_infos[ICI_Node] = clist->maxNbConnectedItem(ItemInternalConnectivityList::NODE_IDX);
  m_infos[ICI_Edge] = clist->maxNbConnectedItem(ItemInternalConnectivityList::EDGE_IDX);
  m_infos[ICI_Face] = clist->maxNbConnectedItem(ItemInternalConnectivityList::FACE_IDX);
  m_infos[ICI_Cell] = clist->maxNbConnectedItem(ItemInternalConnectivityList::CELL_IDX);

  m_infos[ICI_NodeItemTypeInfo] = item_shared_infos->maxLocalNodePerItemType();
  m_infos[ICI_EdgeItemTypeInfo] = item_shared_infos->maxLocalEdgePerItemType();
  m_infos[ICI_FaceItemTypeInfo] = item_shared_infos->maxLocalFacePerItemType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityInfo::
reduce(IParallelMng* pm)
{
  pm->reduce(Parallel::ReduceMax,IntegerArrayView(NB_ICI,m_infos));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
