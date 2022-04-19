// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/IItemFamily.h"

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
class ARCANE_CORE_EXPORT IndexedItemConnectivityViewBase
{
 public:
  //! Nombre d'entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbItem(ItemLocalId lid) const { return m_nb_item[lid]; }
  //! Liste des entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewT<Item> items(ItemLocalId lid) const
  {
    const Int32* ptr = & m_list_data[m_indexes[lid]];
    return { reinterpret_cast<const ItemLocalId*>(ptr), m_nb_item[lid] };
  }
  eItemKind sourceItemKind() const { return m_source_kind; }
  eItemKind targetItemKind() const { return m_target_kind; }
 public:
  //! Initialise la vue
  void init(SmallSpan<const Int32> nb_item,SmallSpan<const Int32> indexes,
            SmallSpan<const Int32> list_data,eItemKind source_kind,eItemKind target_kind)
  {
    m_indexes = indexes;
    m_nb_item = nb_item;
    m_list_data = list_data;
    m_source_kind = source_kind;
    m_target_kind = target_kind;
  }

  void set(IndexedItemConnectivityViewBase view)
  {
    m_indexes = view.m_indexes;
    m_nb_item = view.m_nb_item;
    m_list_data = view.m_list_data;
    m_source_kind = view.m_source_kind;
    m_target_kind = view.m_target_kind;
  }
 protected:
  SmallSpan<const Int32> m_nb_item;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_list_data;
  eItemKind m_source_kind = IK_Unknown;
  eItemKind m_target_kind = IK_Unknown;
 protected:
  [[noreturn]] void _badConversion(eItemKind k1,eItemKind k2) const;
  inline void _checkValid(eItemKind k1,eItemKind k2) const
  {
    if (k1!=m_source_kind || k2!=m_target_kind)
      _badConversion(k1,k2);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue spécialisée sur une connectivité non structurée entre deux entités.
 */
template<typename ItemType1,typename ItemType2>
class IndexedItemConnectivityGenericViewT
: public IndexedItemConnectivityViewBase
{
 public:
  using ItemType1Type = ItemType1;
  using ItemType2Type = ItemType2;
  using ItemLocalId1 = typename ItemType1::LocalIdType;
  using ItemLocalId2 = typename ItemType2::LocalIdType;
  using ItemLocalIdViewType = ItemLocalIdViewT<ItemType2>;
 public:
  IndexedItemConnectivityGenericViewT(IndexedItemConnectivityViewBase view)
  : IndexedItemConnectivityViewBase(view)
  {
#ifdef ARCANE_CHECK
    eItemKind k1 = ItemTraitsT<ItemType1>::kind();
    eItemKind k2 = ItemTraitsT<ItemType2>::kind();
    _checkValid(k1,k2);
#endif
  }
  IndexedItemConnectivityGenericViewT() = default;
 public:
  //! Liste des entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType items(ItemLocalId1 lid) const
  {
    const Int32* ptr = & m_list_data[m_indexes[lid]];
    return { reinterpret_cast<const ItemLocalId2*>(ptr), m_nb_item[lid] };
  }
  //! Liste des entités connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType itemIds(ItemLocalId1 lid) const
  {
    const Int32* ptr = & m_list_data[m_indexes[lid]];
    return { reinterpret_cast<const ItemLocalId2*>(ptr), m_nb_item[lid] };
  }
  //! i-ème entitée connectée à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 itemId(ItemLocalId1 lid,Int32 index) const
  {
    return ItemLocalId2(m_list_data[m_indexes[lid]+index]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Node.
 */
template<typename ItemType>
class IndexedItemConnectivityViewT<ItemType,Node>
: public IndexedItemConnectivityGenericViewT<ItemType,Node>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericViewT<ItemType,Node>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityViewT(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityViewT() = default;
 public:
  //! Nombre de noeuds connectés à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbNode(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des noeuds connectés à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType nodes(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! Liste des noeuds connectés à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType nodeIds(ItemLocalIdType lid) const { return BaseClass::itemIds(lid); }
  //! i-ème noeud connecté à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 nodeId(ItemLocalIdType lid,Int32 index) const { return BaseClass::itemId(lid,index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Edge.
 */
template<typename ItemType>
class IndexedItemConnectivityViewT<ItemType,Edge>
: public IndexedItemConnectivityGenericViewT<ItemType,Edge>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericViewT<ItemType,Edge>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityViewT(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityViewT() = default;
 public:
  //! Nombre d'arêtes connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbEdge(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des arêtes connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType edges(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! Liste des arêtes connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType edgeIds(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! i-ème arête connectée à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 edgeId(ItemLocalIdType lid,Int32 index) const { return BaseClass::itemId(lid,index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Face.
 */
template<typename ItemType>
class IndexedItemConnectivityViewT<ItemType,Face>
: public IndexedItemConnectivityGenericViewT<ItemType,Face>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericViewT<ItemType,Face>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityViewT(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityViewT() = default;
 public:
  //! Nombre de faces connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbFace(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des faces connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType faces(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! Liste des faces connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType faceIds(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! i-ème face connectée à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 faceId(ItemLocalIdType lid,Int32 index) const { return BaseClass::itemId(lid,index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Cell.
 */
template<typename ItemType>
class IndexedItemConnectivityViewT<ItemType,Cell>
: public IndexedItemConnectivityGenericViewT<ItemType,Cell>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericViewT<ItemType,Cell>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityViewT(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityViewT() = default;
 public:
  //! Nombre de mailles connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbCell(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des mailles connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType cells(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! Liste des mailles connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType cellIds(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! i-ème maille connectée à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 cellId(ItemLocalIdType lid,Int32 index) const { return BaseClass::itemId(lid,index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une connectivité ItemType->Dof.
 */
template<typename ItemType>
class IndexedItemConnectivityViewT<ItemType,DoF>
: public IndexedItemConnectivityGenericViewT<ItemType,DoF>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericViewT<ItemType,DoF>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityViewT(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityViewT() = default;
 public:
  //! Nombre de DoFs connectés à l'entité \a lid
  ARCCORE_HOST_DEVICE Int32 nbDof(ItemLocalIdType lid) const { return BaseClass::nbItem(lid); }
  //! Liste des DoFs connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType dofs(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! Liste des DoFs connectées à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalIdViewType dofIds(ItemLocalIdType lid) const { return BaseClass::items(lid); }
  //! i-ème DoF connecté à l'entité \a lid
  ARCCORE_HOST_DEVICE ItemLocalId2 dofId(ItemLocalIdType lid,Int32 index) const { return BaseClass::itemId(lid,index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using IndexedCellNodeConnectivityView = IndexedItemConnectivityViewT<Cell,Node>;
using IndexedCellEdgeConnectivityView = IndexedItemConnectivityViewT<Cell,Edge>;
using IndexedCellFaceConnectivityView = IndexedItemConnectivityViewT<Cell,Face>;
using IndexedCellCellConnectivityView = IndexedItemConnectivityViewT<Cell,Cell>;
using IndexedCellDoFConnectivityView = IndexedItemConnectivityViewT<Cell,DoF>;

using IndexedFaceNodeConnectivityView = IndexedItemConnectivityViewT<Face,Node>;
using IndexedFaceEdgeConnectivityView = IndexedItemConnectivityViewT<Face,Edge>;
using IndexedFaceFaceConnectivityView = IndexedItemConnectivityViewT<Face,Face>;
using IndexedFaceCellConnectivityView = IndexedItemConnectivityViewT<Face,Cell>;
using IndexedFaceDoFConnectivityView = IndexedItemConnectivityViewT<Face,DoF>;

using IndexedEdgeNodeConnectivityView = IndexedItemConnectivityViewT<Edge,Node>;
using IndexedEdgeEdgeConnectivityView = IndexedItemConnectivityViewT<Edge,Edge>;
using IndexedEdgeFaceConnectivityView = IndexedItemConnectivityViewT<Edge,Face>;
using IndexedEdgeCellConnectivityView = IndexedItemConnectivityViewT<Edge,Cell>;
using IndexedEdgeDoFConnectivityView = IndexedItemConnectivityViewT<Edge,DoF>;

using IndexedNodeNodeConnectivityView = IndexedItemConnectivityViewT<Node,Node>;
using IndexedNodeEdgeConnectivityView = IndexedItemConnectivityViewT<Node,Edge>;
using IndexedNodeFaceConnectivityView = IndexedItemConnectivityViewT<Node,Face>;
using IndexedNodeCellConnectivityView = IndexedItemConnectivityViewT<Node,Cell>;
using IndexedNodeDoFConnectivityView = IndexedItemConnectivityViewT<Node,DoF>;

using IndexedDoFDoFConnectivityView = IndexedItemConnectivityViewT<DoF,DoF>;
using IndexedDoFEdgeConnectivityView = IndexedItemConnectivityViewT<DoF,Edge>;
using IndexedDoFFaceConnectivityView = IndexedItemConnectivityViewT<DoF,Face>;
using IndexedDoFCellConnectivityView = IndexedItemConnectivityViewT<DoF,Cell>;
using IndexedDoFDoFConnectivityView = IndexedItemConnectivityViewT<DoF,DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
