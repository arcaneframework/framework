// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshConnectivity.cc                             (C) 2000-2021 */
/*                                                                           */
/* Informations de connectivité d'un maillage non structuré.                 */
/*---------------------------------------------------------------------------*/

#include "arcane/UnstructuredMeshConnectivity.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

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
  ItemInternalConnectivityList* clist = family->_unstructuredItemInternalConnectivityList();
  auto item_index_type = _IDX((ItemType2*)nullptr);
  auto indexes = clist->connectivityIndex(item_index_type);
  auto list_type = clist->connectivityList(item_index_type);
  auto nb_item = clist->connectivityNbItem(item_index_type);
  cview.init(nb_item,indexes,list_type,ik1,ik2);
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
