// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de DynamicMesh.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INTERNAL_DYNAMICMESHINTERNAL_H
#define ARCANE_MESH_INTERNAL_DYNAMICMESHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SmallArray.h"

#include "arcane/core/internal/IMeshInternal.h"
#include "arcane/core/internal/IMeshModifierInternal.h"

#include "arcane/mesh/ItemConnectivityMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshInternal
: public IMeshInternal
, public IMeshModifierInternal
{
 public:

  explicit DynamicMeshInternal(DynamicMesh* mesh);

 public:

  void build();

 public:

  void setMeshKind(const MeshKind& v) override;
  IItemConnectivityMng* dofConnectivityMng() const noexcept override;
  IPolyhedralMeshModifier* polyhedralMeshModifier() const noexcept override;
  void removeNeedRemoveMarkedItems() override;
  NodeLocalId addNode(ItemUniqueId unique_id) override;
  FaceLocalId addFace(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) override;
  CellLocalId addCell(ItemUniqueId unique_id, ItemTypeId type_id, ConstArrayView<Int64> nodes_uid) override;

 private:

  DynamicMesh* m_mesh = nullptr;
  std::unique_ptr<IItemConnectivityMng> m_connectivity_mng = nullptr;
  SmallArray<Int64> m_items_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
