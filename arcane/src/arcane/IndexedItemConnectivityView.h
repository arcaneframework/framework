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
  ARCCORE_HOST_DEVICE ItemLocalIdView<Item> items(ItemLocalId lid) const
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
class IndexedItemConnectivityGenericView
: public IndexedItemConnectivityViewBase
{
 public:
  using ItemType1Type = ItemType1;
  using ItemType2Type = ItemType2;
  using ItemLocalId1 = typename ItemType1::LocalIdType;
  using ItemLocalId2 = typename ItemType2::LocalIdType;
  using ItemLocalIdViewType = ItemLocalIdView<ItemType2>;
 public:
  IndexedItemConnectivityGenericView(IndexedItemConnectivityViewBase view)
  : IndexedItemConnectivityViewBase(view)
  {
#ifdef ARCANE_CHECK
    eItemKind k1 = ItemTraitsT<ItemType1>::kind();
    eItemKind k2 = ItemTraitsT<ItemType2>::kind();
    _checkValid(k1,k2);
#endif
  }
  IndexedItemConnectivityGenericView() = default;
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
class IndexedItemConnectivityView<ItemType,Node>
: public IndexedItemConnectivityGenericView<ItemType,Node>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Node>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityView(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityView() = default;
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
class IndexedItemConnectivityView<ItemType,Edge>
: public IndexedItemConnectivityGenericView<ItemType,Edge>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Edge>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityView(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityView() = default;
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
class IndexedItemConnectivityView<ItemType,Face>
: public IndexedItemConnectivityGenericView<ItemType,Face>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Face>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityView(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityView() = default;
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
class IndexedItemConnectivityView<ItemType,Cell>
: public IndexedItemConnectivityGenericView<ItemType,Cell>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,Cell>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityView(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityView() = default;
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
class IndexedItemConnectivityView<ItemType,DoF>
: public IndexedItemConnectivityGenericView<ItemType,DoF>
{
 public:
  using BaseClass = IndexedItemConnectivityGenericView<ItemType,DoF>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ItemLocalIdViewType = typename BaseClass::ItemLocalIdViewType;
  using ItemLocalId2 = typename BaseClass::ItemLocalId2;
 public:
  IndexedItemConnectivityView(IndexedItemConnectivityViewBase view) : BaseClass(view){}
  IndexedItemConnectivityView() = default;
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

using IndexedCellNodeConnectivityView = IndexedItemConnectivityView<Cell,Node>;
using IndexedCellEdgeConnectivityView = IndexedItemConnectivityView<Cell,Edge>;
using IndexedCellFaceConnectivityView = IndexedItemConnectivityView<Cell,Face>;
using IndexedCellCellConnectivityView = IndexedItemConnectivityView<Cell,Cell>;
using IndexedCellDoFConnectivityView = IndexedItemConnectivityView<Cell,DoF>;

using IndexedFaceNodeConnectivityView = IndexedItemConnectivityView<Face,Node>;
using IndexedFaceEdgeConnectivityView = IndexedItemConnectivityView<Face,Edge>;
using IndexedFaceFaceConnectivityView = IndexedItemConnectivityView<Face,Face>;
using IndexedFaceCellConnectivityView = IndexedItemConnectivityView<Face,Cell>;
using IndexedFaceDoFConnectivityView = IndexedItemConnectivityView<Face,DoF>;

using IndexedEdgeNodeConnectivityView = IndexedItemConnectivityView<Edge,Node>;
using IndexedEdgeEdgeConnectivityView = IndexedItemConnectivityView<Edge,Edge>;
using IndexedEdgeFaceConnectivityView = IndexedItemConnectivityView<Edge,Face>;
using IndexedEdgeCellConnectivityView = IndexedItemConnectivityView<Edge,Cell>;
using IndexedEdgeDoFConnectivityView = IndexedItemConnectivityView<Edge,DoF>;

using IndexedNodeNodeConnectivityView = IndexedItemConnectivityView<Node,Node>;
using IndexedNodeEdgeConnectivityView = IndexedItemConnectivityView<Node,Edge>;
using IndexedNodeFaceConnectivityView = IndexedItemConnectivityView<Node,Face>;
using IndexedNodeCellConnectivityView = IndexedItemConnectivityView<Node,Cell>;
using IndexedNodeDoFConnectivityView = IndexedItemConnectivityView<Node,DoF>;

using IndexedDoFDoFConnectivityView = IndexedItemConnectivityView<DoF,DoF>;
using IndexedDoFEdgeConnectivityView = IndexedItemConnectivityView<DoF,Edge>;
using IndexedDoFFaceConnectivityView = IndexedItemConnectivityView<DoF,Face>;
using IndexedDoFCellConnectivityView = IndexedItemConnectivityView<DoF,Cell>;
using IndexedDoFDoFConnectivityView = IndexedItemConnectivityView<DoF,DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
