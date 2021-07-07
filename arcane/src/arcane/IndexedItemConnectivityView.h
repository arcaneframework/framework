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
 *
 * Comme toute les vues, les instances de cette classe sont temporaires et
 * ne doivent pas être conservées entre deux évolutions du maillage.
 */
class ARCANE_CORE_EXPORT IndexedItemConnectivityBaseView
{
 public:
  //! Nombre d'entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbItem(ItemLocalId lid) const { return m_nb_item[lid]; }
  //! Liste des entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdView<Item> items(ItemLocalId lid) const
  {
    const Int32* ptr = & m_list_data[m_indexes[lid]];
    return { reinterpret_cast<const ItemLocalId*>(ptr), m_nb_item[lid] };
  }
 public:
  //! Initialise la vue
  void init(SmallSpan<const Int32> nb_item,SmallSpan<const Int32> indexes,
            SmallSpan<const Int32> list_data)
  {
    m_indexes = indexes;
    m_nb_item = nb_item;
    m_list_data = list_data;
  }
 protected:
  SmallSpan<const Int32> m_nb_item;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_list_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue spécialisée sur une connectivité non structurée entre deux entités.
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
  //! Liste des entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType items(ItemLocalId1 lid) const
  {
    const Int32* ptr = & m_list_data[m_indexes[lid]];
    return { reinterpret_cast<const ItemLocalId2*>(ptr), m_nb_item[lid] };
  }
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
  //! Nombre de noeuds connectés à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbNode(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des noeuds connectés à l'entité \a lid
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
  //! Nombre d'arêtes connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbEdge(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des arêtes connectées à l'entité \a lid
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
  //! Nombre de faces connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbFace(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des faces connectées à l'entité \a lid
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
  //! Nombre de mailles connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbCell(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des mailles connectées à l'entité \a lid
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
