// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityInfo.cc                                     (C) 2000-2013 */
/*                                                                           */
/* Informations sur la connectivité par type d'entité.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/mesh/ItemConnectivityInfo.h"
#include "arcane/mesh/ItemSharedInfoList.h"

#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

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
fill(ItemSharedInfoList* item_shared_infos)
{
  m_infos[ICI_Node] = item_shared_infos->maxNodePerItem();
  m_infos[ICI_Edge] = item_shared_infos->maxEdgePerItem();
  m_infos[ICI_Face] = item_shared_infos->maxFacePerItem();
  m_infos[ICI_Cell] = item_shared_infos->maxCellPerItem();
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

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
