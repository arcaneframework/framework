// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedItemConnectivityView.h                               (C) 2000-2021 */
/*                                                                           */
/* Vues sur les connectivités utilisant des index.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_INDEXEDITEMCONNECTIVITYVIEW_H
#define ARCANE_INDEXEDITEMCONNECTIVITYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une vue sur une connectivité non structurée.
 */
class ARCANE_CORE_EXPORT IndexedItemConnectivityBaseView
{
 public:
  ARCCORE_HOST_DEVICE Int32 nbItem(ItemLocalId lid) const { return m_nb_item[lid]; }
 protected:
  SmallSpan<const Int32> m_nb_item;
  SmallSpan<const Int32> m_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité non structurée entre deux entités.
 */
template<typename ItemType1,typename ItemType2>
class IndexedItemConnectivityGenericView
: public IndexedItemConnectivityBaseView
{
 public:
  using ItemType1Type = ItemType1;
  using ItemType2Type = ItemType2;
  using ItemLocalId1 = typename ItemType1::LocalIdType;
  using ItemLocalId2 = typename ItemType2::LocalIdType;
  using ItemLocalIdViewType = ItemLocalIdView<ItemType2>;
 public:
  void init(SmallSpan<const Int32> nb_item,SmallSpan<const Int32> indexes,
            SmallSpan<const Int32> list_data)
  {
    m_indexes = indexes;
    m_nb_item = nb_item;
    m_list = { reinterpret_cast<const ItemLocalId2*>(list_data.data()), list_data.size() };
  }
 public:
  ARCCORE_HOST_DEVICE ItemLocalIdViewType items(ItemLocalId1 lid) const
  {
    return { m_list.data() + m_indexes[lid], m_nb_item[lid] };
  }
 protected:
  ItemLocalIdViewType m_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Node.
 */
template<typename ItemType>
class IndexedItemConnectivityView<ItemType,Node>
: public IndexedItemConnectivityGenericView<ItemType,Node>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Node>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
 public:
  ARCCORE_HOST_DEVICE Int32 nbNode(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  ARCCORE_HOST_DEVICE ItemLocalIdViewType nodes(ItemLocalIdType lid) const { return BaseClass::items(lid); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Edge.
 */
template<typename ItemType>
class IndexedItemConnectivityView<ItemType,Edge>
: public IndexedItemConnectivityGenericView<ItemType,Edge>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Edge>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
 public:
  ARCCORE_HOST_DEVICE Int32 nbEdge(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  ARCCORE_HOST_DEVICE ItemLocalIdViewType edges(ItemLocalIdType lid) const { return BaseClass::items(lid); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Face.
 */
template<typename ItemType>
class IndexedItemConnectivityView<ItemType,Face>
: public IndexedItemConnectivityGenericView<ItemType,Face>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Face>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
 public:
  ARCCORE_HOST_DEVICE Int32 nbFace(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  ARCCORE_HOST_DEVICE ItemLocalIdViewType faces(ItemLocalIdType lid) const { return BaseClass::items(lid); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Cell.
 */
template<typename ItemType>
class IndexedItemConnectivityView<ItemType,Cell>
: public IndexedItemConnectivityGenericView<ItemType,Cell>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Cell>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
 public:
  ARCCORE_HOST_DEVICE Int32 nbCell(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  ARCCORE_HOST_DEVICE ItemLocalIdViewType cells(ItemLocalIdType lid) const { return BaseClass::items(lid); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using IndexedCellNodeConnectivityView = IndexedItemConnectivityView<Cell,Node>;
using IndexedCellEdgeConnectivityView = IndexedItemConnectivityView<Cell,Edge>;
using IndexedCellFaceConnectivityView = IndexedItemConnectivityView<Cell,Face>;

using IndexedFaceNodeConnectivityView = IndexedItemConnectivityView<Face,Node>;
using IndexedFaceEdgeConnectivityView = IndexedItemConnectivityView<Face,Edge>;
using IndexedFaceCellConnectivityView = IndexedItemConnectivityView<Face,Cell>;

using IndexedNodeEdgeConnectivityView = IndexedItemConnectivityView<Node,Edge>;
using IndexedNodeFaceConnectivityView = IndexedItemConnectivityView<Node,Face>;
using IndexedNodeCellConnectivityView = IndexedItemConnectivityView<Node,Cell>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
