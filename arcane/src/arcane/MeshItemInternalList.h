// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshItemInternalList.h                                      (C) 2000-2022 */
/*                                                                           */
/* Tableaux d'indirection sur les entités d'un maillage.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHITEMINTERNALLIST_H
#define ARCANE_MESHITEMINTERNALLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;
class ItemInternalConnectivityList;

namespace mesh
{
class DynamicMesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableaux d'indirection sur les entités d'un maillage.
 */
class MeshItemInternalList
{
  friend class mesh::DynamicMesh;
  friend class ItemInternalConnectivityList;

 public:

  ItemInternalArrayView nodes;
  ItemInternalArrayView edges;
  ItemInternalArrayView faces;
  ItemInternalArrayView cells;
  IMesh* mesh = nullptr;

 private:

  ItemSharedInfo* m_node_shared_info = nullptr;
  ItemSharedInfo* m_edge_shared_info = nullptr;
  ItemSharedInfo* m_face_shared_info = nullptr;
  ItemSharedInfo* m_cell_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

