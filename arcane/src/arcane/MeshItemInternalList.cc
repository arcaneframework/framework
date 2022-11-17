// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshItemInternalList.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Tableaux d'indirection sur les entités d'un maillage.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshItemInternalList.h"

#include "arcane/ItemSharedInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshItemInternalList::
_internalSetNodeSharedInfo(ItemSharedInfo* s)
{
  m_node_shared_info = s;
  _notifyUpdate();
}

void MeshItemInternalList::
_internalSetEdgeSharedInfo(ItemSharedInfo* s)
{
  m_edge_shared_info = s;
  _notifyUpdate();
}

void MeshItemInternalList::
_internalSetFaceSharedInfo(ItemSharedInfo* s)
{
  m_face_shared_info = s;
  _notifyUpdate();
}

void MeshItemInternalList::
_internalSetCellSharedInfo(ItemSharedInfo* s)
{
  m_cell_shared_info = s;
  _notifyUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshItemInternalList::
_notifyUpdate()
{
  if (m_node_shared_info)
    m_node_shared_info->updateMeshItemInternalList();
  if (m_edge_shared_info)
    m_edge_shared_info->updateMeshItemInternalList();
  if (m_face_shared_info)
    m_face_shared_info->updateMeshItemInternalList();
  if (m_cell_shared_info)
    m_cell_shared_info->updateMeshItemInternalList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
