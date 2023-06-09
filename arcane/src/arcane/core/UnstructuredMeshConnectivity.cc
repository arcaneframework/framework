// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshConnectivity.cc                             (C) 2000-2023 */
/*                                                                           */
/* Informations de connectivité d'un maillage non structuré.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

inline constexpr Int32 _IDX(Node*) { return ItemInternalConnectivityList::NODE_IDX; }
inline constexpr Int32 _IDX(Edge*) { return ItemInternalConnectivityList::EDGE_IDX; }
inline constexpr Int32 _IDX(Face*) { return ItemInternalConnectivityList::FACE_IDX; }
inline constexpr Int32 _IDX(Cell*) { return ItemInternalConnectivityList::CELL_IDX; }

template<typename ConnectivityView> inline void
_internalInit(ConnectivityView& cview,IMesh* mesh)
{
  using ItemType1 = typename ConnectivityView::ItemType1Type;
  using ItemType2 = typename ConnectivityView::ItemType2Type;

  eItemKind ik1 = ItemTraitsT<ItemType1>::kind();
  eItemKind ik2 = ItemTraitsT<ItemType2>::kind();

  IItemFamily* family = mesh->itemFamily(ik1);
  ItemInternalConnectivityList* clist = family->_internalApi()->unstructuredItemInternalConnectivityList();
  auto item_index_type = _IDX((ItemType2*)nullptr);
  auto container_view = clist->containerView(item_index_type);
  cview = IndexedItemConnectivityViewBase(container_view,ik1,ik2);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshConnectivityView::
setMesh(IMesh* mesh)
{
  m_mesh = mesh;

  _internalInit(m_cell_node_connectivity_view,mesh);
  _internalInit(m_cell_edge_connectivity_view,mesh);
  _internalInit(m_cell_face_connectivity_view,mesh);

  _internalInit(m_face_node_connectivity_view,mesh);
  _internalInit(m_face_edge_connectivity_view,mesh);
  _internalInit(m_face_cell_connectivity_view,mesh);

  _internalInit(m_node_edge_connectivity_view,mesh);
  _internalInit(m_node_face_connectivity_view,mesh);
  _internalInit(m_node_cell_connectivity_view,mesh);

  _internalInit(m_edge_node_connectivity_view,mesh);
  _internalInit(m_edge_face_connectivity_view,mesh);
  _internalInit(m_edge_cell_connectivity_view,mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshConnectivityView::
_checkValid() const
{
  if (!m_mesh)
    ARCANE_FATAL("Can not use unitialised UnstructuredMeshConnectivityView.\n"
                 "Call the method setMesh() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
