// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshItemInternalList.h                                      (C) 2000-2024 */
/*                                                                           */
/* Tableaux d'indirection sur les entités d'un maillage.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHITEMINTERNALLIST_H
#define ARCANE_CORE_MESHITEMINTERNALLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;
class ItemInternalConnectivityList;
}

namespace Arcane::mesh
{
class DynamicMesh;
class PolyhedralMesh;
}

namespace Arcane::impl
{
/*!
 * \internal
 * \brief Liste des ItemSharedInfo associés à un maillage.
 */
class MeshItemSharedInfoList
{
  friend ItemInternalConnectivityList;
  friend ItemBase;

 private:

  MeshItemSharedInfoList() = default;
  MeshItemSharedInfoList(ItemSharedInfo* v)
  : m_node(v)
  , m_edge(v)
  , m_face(v)
  , m_cell(v)
  {}

 private:

  ItemSharedInfo* m_node = nullptr;
  ItemSharedInfo* m_edge = nullptr;
  ItemSharedInfo* m_face = nullptr;
  ItemSharedInfo* m_cell = nullptr;
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableaux d'indirection sur les entités d'un maillage.
 */
class ARCANE_CORE_EXPORT MeshItemInternalList
{
  // Les deux classes suivantes ont besoin d'accéder aux
  // méthodes _internalSet*().
  friend class mesh::DynamicMesh;
  friend class mesh::PolyhedralMesh;

  friend class ItemInternalConnectivityList;
  friend class ItemBase;

 public:

  ItemInternalArrayView nodes;
  ItemInternalArrayView edges;
  ItemInternalArrayView faces;
  ItemInternalArrayView cells;
  IMesh* mesh = nullptr;

 private:

  void _internalSetNodeSharedInfo(ItemSharedInfo* s);
  void _internalSetEdgeSharedInfo(ItemSharedInfo* s);
  void _internalSetFaceSharedInfo(ItemSharedInfo* s);
  void _internalSetCellSharedInfo(ItemSharedInfo* s);

 private:

  // Ne pas modifier directement ces champs.
  // Utiliser les méthodes _internalSet*() correspondantes
  ItemSharedInfo* m_node_shared_info = nullptr;
  ItemSharedInfo* m_edge_shared_info = nullptr;
  ItemSharedInfo* m_face_shared_info = nullptr;
  ItemSharedInfo* m_cell_shared_info = nullptr;

 private:

  void _notifyUpdate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

