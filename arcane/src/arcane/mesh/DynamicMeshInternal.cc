// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshInternal.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de DynamicMesh.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/internal/DynamicMeshInternal.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/ItemConnectivityMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshInternal::
DynamicMeshInternal(DynamicMesh* mesh)
: m_mesh(mesh)
, m_connectivity_mng(std::make_unique<ItemConnectivityMng>(mesh->traceMng()))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshInternal::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshInternal::
setMeshKind(const MeshKind& v)
{
  m_mesh->m_mesh_kind = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemConnectivityMng* DynamicMeshInternal::
dofConnectivityMng() const noexcept
{
  return m_connectivity_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPolyhedralMeshModifier* DynamicMeshInternal::
polyhedralMeshModifier() const noexcept
{
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshInternal::
removeNeedRemoveMarkedItems()
{
  m_mesh->incrementalBuilder()->removeNeedRemoveMarkedItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeLocalId DynamicMeshInternal::
addNode([[maybe_unused]] ItemUniqueId unique_id)
{
  ARCANE_THROW(NotImplementedException, "");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceLocalId DynamicMeshInternal::
addFace([[maybe_unused]] ItemUniqueId unique_id,
        [[maybe_unused]] ItemTypeId type_id,
        [[maybe_unused]] ConstArrayView<Int64> nodes_uid)
{
  ARCANE_THROW(NotImplementedException, "");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellLocalId DynamicMeshInternal::
addCell(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid)
{
  Int32 nb_node = nodes_uid.size();
  m_items_infos.resize(nb_node + 2);
  m_items_infos[0] = type_id;
  m_items_infos[1] = unique_id;
  m_items_infos.subView(2, nb_node).copy(nodes_uid);
  Int32 cell_local_id = NULL_ITEM_LOCAL_ID;
  m_mesh->addCells(1, m_items_infos, ArrayView<Int32>(1,&cell_local_id));
  return CellLocalId(cell_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
