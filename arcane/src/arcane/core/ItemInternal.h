// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.h                                              (C) 2000-2024 */
/*                                                                           */
/* Partie interne d'une entité.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINTERNAL_H
#define ARCANE_CORE_ITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemIndexedListView.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemUniqueId.h"
#include "arcane/core/ItemLocalIdListView.h"
#include "arcane/core/ItemTypeId.h"
#include "arcane/core/ItemFlags.h"
#include "arcane/core/ItemConnectivityContainerView.h"
#include "arcane/core/ItemInternalVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef null
#undef null
#endif

//#define ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO

#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
#define A_INTERNAL_SI(name) m_shared_infos.m_##name
#else
#define A_INTERNAL_SI(name) m_items->m_##name##_shared_info
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class IncrementalItemConnectivityBase;
class PolyhedralFamily;
class PolyhedralMeshImpl;
class FaceFamily;
class MeshRefinement;
}
namespace Arcane::Materials
{
class ConstituentItemSharedInfo;
}
namespace Arcane
{
class ItemInternalCompatibility;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour construire une instance de ItemBase
 */
class ARCANE_CORE_EXPORT ItemBaseBuildInfo
{
 public:
  ItemBaseBuildInfo() = default;
  ItemBaseBuildInfo(Int32 local_id,ItemSharedInfo* shared_info)
  : m_local_id(local_id), m_shared_info(shared_info) {}
 public:
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
  ItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations de connectivité, pour une famille d'entité,
 * permettant la transition entre les anciennes et nouvelles implémentations
 * des connectivités.
 */
class ARCANE_CORE_EXPORT ItemInternalConnectivityList
{
  // IMPORTANT: Cette structure doit avoir le même agencement mémoire
  // que la structure C# de même nom.

  friend class ItemBase;
  friend class ItemInternal;
  friend class Item;

  // Pour accès à _setConnectivity*
  friend mesh::IncrementalItemConnectivityBase;
  friend mesh::PolyhedralFamily;
  friend mesh::PolyhedralMeshImpl;

  // Pour accès à m_items
  friend mesh::ItemFamily;

 private:

  /*!
   * \brief Vue spécifique pour gérer les entités nulles.
   *
   * Pour l'entité nulle, l'index vaut NULL_ITEM_LOCAL_ID (soit (-1)) et il faut
   * pouvoir accéder à `m_data` avec cet indice ce qui n'est pas possible
   * avec le ArrayView classique en mode check.
   */
  struct Int32View
  {
   public:
    Int32 operator[](Int32 index) const
    {
#ifdef ARCANE_CHECK
      if (index==NULL_ITEM_LOCAL_ID){
        // Pour l'entité nulle, la taille doit être 0.
        if (m_size!=0)
          arcaneRangeError(index,m_size);
      }
      else
        ARCANE_CHECK_AT(index,m_size);
#endif
      return m_data[index];
    }
    void operator=(ConstArrayView<Int32> v)
    {
      m_data = v.data();
      m_size = v.size();
    }
    // Il faut que data[NULL_ITEM_LOCAL_ID] soit valide.
    // Il faut donc que (data-1) pointe vers une adresse valide
    void setNull(const Int32* data)
    {
      m_data = data;
      m_size = 0;
    }
    operator ConstArrayView<Int32>() const
    {
      return ConstArrayView<Int32>(m_size,m_data);
    }
    operator SmallSpan<const Int32>() const
    {
      return SmallSpan<const Int32>(m_data,m_size);
    }
   private:
    Int32 m_size;
    const Int32* m_data;
  };

 public:

  enum
  {
    NODE_IDX = 0,
    EDGE_IDX = 1,
    FACE_IDX = 2,
    CELL_IDX = 3,
    HPARENT_IDX = 4,
    HCHILD_IDX = 5,
    MAX_ITEM_KIND = 6
  };

 public:

  static ItemInternalConnectivityList nullInstance;

 public:

  ItemInternalConnectivityList()
  : m_items(nullptr)
  {
    for( Integer i=0; i<MAX_ITEM_KIND; ++i ){
      m_kind_info[i].m_nb_item_null_data[0] = 0;
      m_kind_info[i].m_nb_item_null_data[1] = 0;
      m_kind_info[i].m_max_nb_item = 0;
    }

    for( Integer i=0; i<MAX_ITEM_KIND; ++i ){
      m_container[i].m_nb_item.setNull(&m_kind_info[i].m_nb_item_null_data[1]);
      m_container[i].m_offset = ConstArrayView<Int32>{};
    }
  }

 public:

  void updateMeshItemInternalList()
  {
#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
    m_shared_infos.m_node = m_items->m_node_shared_info;
    m_shared_infos.m_edge = m_items->m_edge_shared_info;
    m_shared_infos.m_face = m_items->m_face_shared_info;
    m_shared_infos.m_cell = m_items->m_cell_shared_info;
#endif
  }

 private:

  /*!
   * \brief localId() de la \a index-ème entité de type \a item_kind
   * connectés à l'entité de de localid() \a lid.
   */
  Int32 itemLocalId(Int32 item_kind,Int32 lid,Integer index) const
  {
    return m_container[item_kind].itemLocalId(lid,index);
  }
  //! Nombre d'appel à itemLocalId()
  Int64 nbAccess() const { return 0; }
  //! Nombre d'appel à itemLocalIds()
  Int64 nbAccessAll() const { return 0; }

 private:

  //! Positionne le tableau d'index des connectivités
  void _setConnectivityIndex(Int32 item_kind,ConstArrayView<Int32> v)
  {
    m_container[item_kind].m_indexes = v;
  }
  //! Positionne le tableau contenant la liste des connectivités
  void _setConnectivityList(Int32 item_kind,ConstArrayView<Int32> v)
  {
    m_container[item_kind].m_list = v;
    m_container[item_kind].m_offset = ConstArrayView<Int32>{};
  }
  //! Positionne le tableau contenant le nombre d'entités connectées.
  void _setConnectivityNbItem(Int32 item_kind,ConstArrayView<Int32> v)
  {
    m_container[item_kind].m_nb_item = v;
  }
  //! Positionne le nombre maximum d'entités connectées.
  void _setMaxNbConnectedItem(Int32 item_kind,Int32 v)
  {
    m_kind_info[item_kind].m_max_nb_item = v;
  }

 public:

  //! Tableau d'index des connectivités pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityIndex(Int32 item_kind) const
  {
    return m_container[item_kind].m_indexes;
  }
  //! Tableau contenant la liste des connectivités pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityList(Int32 item_kind) const
  {
    return m_container[item_kind].m_list;
  }
  //! Tableau contenant le nombre d'entités connectées pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityNbItem(Int32 item_kind) const
  {
    return m_container[item_kind].m_nb_item;
  }

 public:

  //! Nombre maximum d'entités connectées.
  Int32 maxNbConnectedItem(Int32 item_kind) const
  {
    return m_kind_info[item_kind].m_max_nb_item;
  }

  ItemConnectivityContainerView containerView(Int32 item_kind) const
  {
    return m_container[item_kind].containerView();
  }

 public:

  ItemBaseBuildInfo nodeBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_nodeLocalIdV2(lid,aindex),A_INTERNAL_SI(node)); }
  ItemBaseBuildInfo edgeBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_edgeLocalIdV2(lid,aindex),A_INTERNAL_SI(edge)); }
  ItemBaseBuildInfo faceBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_faceLocalIdV2(lid,aindex),A_INTERNAL_SI(face)); }
  ItemBaseBuildInfo cellBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_cellLocalIdV2(lid,aindex),A_INTERNAL_SI(cell)); }
  ItemBaseBuildInfo hParentBase(Int32 lid, Int32 aindex, ItemSharedInfo* isf) const
  {
    return ItemBaseBuildInfo(_hParentLocalIdV2(lid, aindex), isf);
  }
  ItemBaseBuildInfo hChildBase(Int32 lid, Int32 aindex, ItemSharedInfo* isf) const
  {
    return ItemBaseBuildInfo(_hChildLocalIdV2(lid, aindex), isf);
  }

  auto nodeList(Int32 lid) const { return impl::ItemIndexedListView { A_INTERNAL_SI(node),_itemLocalIdListView(NODE_IDX,lid) }; }
  auto edgeList(Int32 lid) const { return impl::ItemIndexedListView { A_INTERNAL_SI(edge),_itemLocalIdListView(EDGE_IDX,lid) }; }
  auto faceList(Int32 lid) const { return impl::ItemIndexedListView { A_INTERNAL_SI(face),_itemLocalIdListView(FACE_IDX,lid) }; }
  auto cellList(Int32 lid) const { return impl::ItemIndexedListView { A_INTERNAL_SI(cell),_itemLocalIdListView(CELL_IDX,lid) }; }

 private:

  // Ces 4 méthodes sont encore utilisées par ItemBase via internalNodes(), internalEdges(), ...
  // On pourra les supprimer quand ces méthodes obsolètes seront supprimées
  ItemInternalVectorView nodesV2(Int32 lid) const { return { A_INTERNAL_SI(node),_itemLocalIdListView(NODE_IDX,lid) }; }
  ItemInternalVectorView edgesV2(Int32 lid) const { return { A_INTERNAL_SI(edge),_itemLocalIdListView(EDGE_IDX,lid) }; }
  ItemInternalVectorView facesV2(Int32 lid) const { return { A_INTERNAL_SI(face),_itemLocalIdListView(FACE_IDX,lid) }; }
  ItemInternalVectorView cellsV2(Int32 lid) const { return { A_INTERNAL_SI(cell),_itemLocalIdListView(CELL_IDX,lid) }; }

  NodeLocalIdView nodeLocalIdsView(Int32 lid) const { return NodeLocalIdView(_itemLocalIdListView(NODE_IDX,lid)); }
  EdgeLocalIdView edgeLocalIdsView(Int32 lid) const { return EdgeLocalIdView(_itemLocalIdListView(EDGE_IDX,lid)); }
  FaceLocalIdView faceLocalIdsView(Int32 lid) const { return FaceLocalIdView(_itemLocalIdListView(FACE_IDX,lid)); }
  CellLocalIdView cellLocalIdsView(Int32 lid) const { return CellLocalIdView(_itemLocalIdListView(CELL_IDX,lid)); }

 private:

  Int32 _nodeLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(NODE_IDX,lid,index); }
  Int32 _edgeLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(EDGE_IDX,lid,index); }
  Int32 _faceLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(FACE_IDX,lid,index); }
  Int32 _cellLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(CELL_IDX,lid,index); }
  Int32 _hParentLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(HPARENT_IDX,lid,index); }
  Int32 _hChildLocalIdV2(Int32 lid,Int32 index) const { return itemLocalId(HCHILD_IDX,lid,index); }

 private:

  ItemInternal* _nodeV2(Int32 lid,Int32 aindex) const { return m_items->nodes[ _nodeLocalIdV2(lid,aindex) ]; }
  ItemInternal* _edgeV2(Int32 lid,Int32 aindex) const { return m_items->edges[ _edgeLocalIdV2(lid,aindex) ]; }
  ItemInternal* _faceV2(Int32 lid,Int32 aindex) const { return m_items->faces[ _faceLocalIdV2(lid,aindex) ]; }
  ItemInternal* _cellV2(Int32 lid,Int32 aindex) const { return m_items->cells[ _cellLocalIdV2(lid,aindex) ]; }
  ItemInternal* _hParentV2(Int32 lid,Int32 aindex) const { return m_items->cells[ _hParentLocalIdV2(lid,aindex) ]; }
  ItemInternal* _hChildV2(Int32 lid,Int32 aindex) const { return m_items->cells[ _hChildLocalIdV2(lid,aindex) ]; }

 private:

  Int32 _nbNodeV2(Int32 lid) const { return m_container[NODE_IDX].m_nb_item[lid]; }
  Int32 _nbEdgeV2(Int32 lid) const { return m_container[EDGE_IDX].m_nb_item[lid]; }
  Int32 _nbFaceV2(Int32 lid) const { return m_container[FACE_IDX].m_nb_item[lid]; }
  Int32 _nbCellV2(Int32 lid) const { return m_container[CELL_IDX].m_nb_item[lid]; }
  Int32 _nbHParentV2(Int32 lid) const { return m_container[HPARENT_IDX].m_nb_item[lid]; }
  Int32 _nbHChildrenV2(Int32 lid) const { return m_container[HCHILD_IDX].m_nb_item[lid]; }

 private:

  Int32 _nodeOffset(Int32 lid) const { return m_container[NODE_IDX].itemOffset(lid); }
  Int32 _edgeOffset(Int32 lid) const { return m_container[EDGE_IDX].itemOffset(lid); }
  Int32 _faceOffset(Int32 lid) const { return m_container[FACE_IDX].itemOffset(lid); }
  Int32 _cellOffset(Int32 lid) const { return m_container[CELL_IDX].itemOffset(lid); }
  Int32 _itemOffset(Int32 item_kind,Int32 lid) const { return m_container[item_kind].itemOffset(lid); }

 private:

  impl::ItemLocalIdListContainerView _itemLocalIdListView(Int32 item_kind,Int32 lid) const
  {
    return m_container[item_kind].itemLocalIdListView(lid);
  }

 private:

  // NOTE : à terme, il faudra fusionner cette classe avec ItemConnectivityContainerView
  //! Conteneur des vues pour les informations de connectivité d'une famille
  struct Container
  {
    impl::ItemLocalIdListContainerView itemLocalIdListView(Int32 lid) const
    {
      return impl::ItemLocalIdListContainerView(itemLocalIdsData(lid),m_nb_item[lid],itemOffset(lid));
    }
    const Int32* itemLocalIdsData(Int32 lid) const
    {
      return &(m_list[ m_indexes[lid] ]);
    }
    Int32 itemLocalId(Int32 lid,Integer index) const
    {
      return m_list[ m_indexes[lid] + index] + itemOffset(lid);
    }
    ItemConnectivityContainerView containerView() const
    {
      return ItemConnectivityContainerView( m_list, m_indexes, m_nb_item );
    }
    Int32 itemOffset([[maybe_unused]] Int32 lid) const
    {
#ifdef ARCANE_USE_OFFSET_FOR_CONNECTIVITY
      return m_offset[lid];
#else
      return 0;
#endif
    }

   public:

    ConstArrayView<Int32> m_indexes;
    Int32View m_nb_item;
    ConstArrayView<Int32> m_list;
    ConstArrayView<Int32> m_offset;
  };

  struct KindInfo
  {
    Int32 m_max_nb_item;
    Int32 m_nb_item_null_data[2];
  };

 private:

  Container m_container[MAX_ITEM_KIND];
  KindInfo m_kind_info[MAX_ITEM_KIND];

  MeshItemInternalList* m_items;

 private:

#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
  impl::MeshItemSharedInfoList m_shared_infos;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour les entités du maillage.
 *
 * Cette classe est interne à %Arcane.
 *
 * Cette classe est normalement interne à Arcane et il est préférable d'utiliser
 * les versions spécialisés telles que Item, Node, Face, Edge, Cell, Particle
 * ou DoF.
 *
 * Les instances de cette classe sont des objets temporaires qui ne doivent
 * pas être conservés entre deux modifications topologiques du maillage s'il y
 * a des compressions (IItemFamily::compactItems()) lors de ces modifications.
 *
 * L'ensemble des méthodes de cette classe sont en lecture seule et ne
 * permettent pas de modifier une entité.
 */
class ARCANE_CORE_EXPORT ItemBase
: public ItemFlags
{
  friend class ::Arcane::ItemInternal;
  friend class ::Arcane::Item;
  friend class ::Arcane::ItemInternalCompatibility;
  friend Arcane::Materials::ConstituentItemSharedInfo;
  friend class ::Arcane::ItemEnumerator;
  friend MutableItemBase;
  // Pour _internalActiveCells2().
  friend class ::Arcane::Node;
  // Pour _itemInternal()
  friend class ::Arcane::mesh::ItemFamily;
  friend class ::Arcane::mesh::MeshRefinement;

 private:

  ItemBase(Int32 local_id,ItemSharedInfo* shared_info)
  : m_local_id(local_id), m_shared_info(shared_info) {}

 public:

  ItemBase() : m_shared_info(ItemSharedInfo::nullItemSharedInfoPointer) {}
  ItemBase(ItemBaseBuildInfo x) : m_local_id(x.m_local_id), m_shared_info(x.m_shared_info) {}

 public:

  // TODO: A supprimer à terme
  inline ItemBase(ItemInternal* x);

 public:

  //! Numéro local (au sous-domaine) de l'entité
  Int32 localId() const { return m_local_id; }
  //! Numéro local (au sous-domaine) de l'entité
  inline ItemLocalId itemLocalId() const;
  //! Numéro unique de l'entité
  ItemUniqueId uniqueId() const
  {
#ifdef ARCANE_CHECK
    if (m_local_id!=NULL_ITEM_LOCAL_ID)
      arcaneCheckAt((Integer)m_local_id,m_shared_info->m_unique_ids.size());
#endif
    // Ne pas utiliser l'accesseur normal car ce tableau peut etre utilise pour la maille
    // nulle et dans ce cas m_local_id vaut NULL_ITEM_LOCAL_ID (qui est negatif)
    // ce qui provoque une exception pour debordement de tableau.
    return ItemUniqueId(m_shared_info->m_unique_ids.data()[m_local_id]);
  }

  //! Numéro du sous-domaine propriétaire de l'entité
  Int32 owner() const { return m_shared_info->_ownerV2(m_local_id); }

  //! Flags de l'entité
  Int32 flags() const { return m_shared_info->_flagsV2(m_local_id); }

  //! Nombre de noeuds de l'entité
  Integer nbNode() const { return _connectivity()->_nbNodeV2(m_local_id); }
  //! Nombre d'arêtes de l'entité ou nombre d'arêtes connectés à l'entités (pour les noeuds)
  Integer nbEdge() const { return _connectivity()->_nbEdgeV2(m_local_id); }
  //! Nombre de faces de l'entité ou nombre de faces connectés à l'entités (pour les noeuds et arêtes)
  Integer nbFace() const { return _connectivity()->_nbFaceV2(m_local_id); }
  //! Nombre de mailles connectées à l'entité (pour les noeuds, arêtes et faces)
  Integer nbCell() const { return _connectivity()->_nbCellV2(m_local_id); }
  //! Nombre de parents pour l'AMR
  Int32 nbHParent() const { return _connectivity()->_nbHParentV2(m_local_id); }
  //! Nombre d'enfants pour l'AMR
  Int32 nbHChildren() const { return _connectivity()->_nbHChildrenV2(m_local_id); }
  //! Nombre de parent pour les sous-maillages
  Integer nbParent() const { return m_shared_info->nbParent(); }

 public:

  //! Type de l'entité
  Int16 typeId() const { return m_shared_info->_typeId(m_local_id); }
  //! Type de l'entité
  ItemTypeId itemTypeId() const { return ItemTypeId(typeId()); }
  //! Type de l'entité.
  ItemTypeInfo* typeInfo() const { return m_shared_info->typeInfoFromId(typeId()); }

  //! @returns le niveau de raffinement de l'item courant. Si l'item
  //! parent est \p NULL donc par convention il est au niveau 0,
  //! sinon il est simplement au niveau que celui de son parent.
  inline Int32 level() const
  {
    //! si je n'ai pas de parent donc j'ai été crée
    //! directement à partir d'un fichier ou par l'utilisateur,
    //! donc je suis un item de niveau 0
    if (this->nbHParent() == 0)
      return 0;
    //! sinon je suis au niveau supérieur que celui de mon parent
    return (this->hParentBase(0).level() + 1);
  }

  //! @returns \p true si l'item est un ancetre (i.e. a un
  //! enfant actif ou un enfant ancetre), \p false sinon. Renvoie toujours \p false si l'AMR est désactivé.
  inline bool isAncestor() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return false;
    if (this->hChildBase(0).isActive())
      return true;
    return this->hChildBase(0).isAncestor();
  }
  //! @returns \p true si l'item a des enfants (actifs ou non),
  //! \p false  sinon. Renvoie toujours \p false si l'AMR est désactivé.
  inline bool hasHChildren () const
  {
    if (this->nbHChildren() == 0) // TODO ? à vérifier !
      return false;
    else
      return true;
  }

  //! @returns \p true si l'item est actif (i.e. n'a pas de
  //! descendants actifs), \p false  sinon. Notez qu'il suffit de vérifier
  //! le premier enfant seulement. Renvoie toujours \p true si l'AMR est désactivé.
  inline bool isActive() const
  {
    if ( (flags() & II_Inactive) | (flags() & II_CoarsenInactive))
      return false;
    else
      return true;
  }

  //! @returns \p true si l'item est subactif (i.e. pas actif et n'a pas de
  //! descendants), \p false  sinon.Renvoie toujours \p false si l'AMR est désactivé.
  inline  bool isSubactive() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return true;
    return this->hChildBase(0).isSubactive();
  }

  //! Famille dont est issue l'entité
  IItemFamily* family() const { return m_shared_info->m_item_family; }
  //! Genre de l'entité
  eItemKind kind() const { return m_shared_info->m_item_kind; }
  //! Vrai si l'entité est l'entité nulle
  bool null() const { return m_local_id==NULL_ITEM_LOCAL_ID; }
  //! Vrai si l'entité est l'entité nulle
  bool isNull() const { return m_local_id==NULL_ITEM_LOCAL_ID; }
  //! Vrai si l'entité appartient au sous-domaine
  bool isOwn() const { return ItemFlags::isOwn(flags()); }
  /*!
   * \brief Vrai si l'entité est partagé d'autres sous-domaines.
   *
   * Cette méthode n'est pertinente que si les informations de connectivités
   * ont été calculées.
   */
  bool isShared() const { return ItemFlags::isShared(flags()); }

  //! Vrai si l'entité est supprimée
  bool isSuppressed() const { return (flags() & II_Suppressed)!=0; }
  //! Vrai si l'entité est détachée
  bool isDetached() const { return (flags() & II_Detached)!=0; }

  //! \a true si l'entité est sur la frontière
  bool isBoundary() const { return ItemFlags::isBoundary(flags()); }
  //! Maille connectée à l'entité si l'entité est une entité sur la frontière (0 si aucune)
  ItemBase boundaryCell() const { return (flags() & II_Boundary) ? cellBase(0) : ItemBase(); }
  //! Maille derrière l'entité (nullItem() si aucune)
  ItemBase backCell() const
  {
    if (flags() & II_HasBackCell)
      return cellBase((flags() & II_BackCellIsFirst) ? 0 : 1);
    return {};
  }
  //! Maille derrière l'entité (NULL_ITEM_LOCAL_ID si aucune)
  Int32 backCellId() const
  {
    if (flags() & II_HasBackCell)
      return cellId((flags() & II_BackCellIsFirst) ? 0 : 1);
    return NULL_ITEM_LOCAL_ID;
  }
  //! Maille devant l'entité (nullItem() si aucune)
  ItemBase frontCell() const
  {
    if (flags() & II_HasFrontCell)
      return cellBase((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return {};
  }
  //! Maille devant l'entité (NULL_ITEM_LOCAL_ID si aucune)
  Int32 frontCellId() const
  {
    if (flags() & II_HasFrontCell)
      return cellId((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return NULL_ITEM_LOCAL_ID;
  }
  ItemBase masterFace() const
  {
    if (flags() & II_SlaveFace)
      return faceBase(0);
    return {};
  }
  //! \a true s'il s'agit de la face maître d'une interface
  inline bool isMasterFace() const { return flags() & II_MasterFace; }

  //! \a true s'il s'agit d'une face esclave d'une interface
  inline bool isSlaveFace() const { return flags() & II_SlaveFace; }

  Int32 parentId(Integer index) const { return m_shared_info->_parentLocalIdV2(m_local_id,index); }

  //@{
  Int32 nodeId(Integer index) const { return _connectivity()->_nodeLocalIdV2(m_local_id,index); }
  Int32 edgeId(Integer index) const { return _connectivity()->_edgeLocalIdV2(m_local_id,index); }
  Int32 faceId(Integer index) const { return _connectivity()->_faceLocalIdV2(m_local_id,index); }
  Int32 cellId(Integer index) const { return _connectivity()->_cellLocalIdV2(m_local_id,index); }
  Int32 hParentId(Int32 index) const { return _connectivity()->_hParentLocalIdV2(m_local_id,index); }
  Int32 hChildId(Int32 index) const { return _connectivity()->_hChildLocalIdV2(m_local_id,index); }
  //@}

  /*!
   * \brief Méthodes utilisant les nouvelles connectivités pour accéder
   * aux informations de connectivité. A ne pas utiliser en dehors de Arcane.
   *
   * \warning Ces méthodes ne doivent être appelées que sur les entités
   * qui possèdent la connectivité associée ET qui sont au nouveau format.
   * Par exemple, cela ne fonctionne pas sur Cell->Cell car il n'y a pas de
   * connectivité maille/maille. En cas de mauvaise utilisation, cela
   * se traduit par un débordement de tableau.
   */
  //@{
  ARCANE_DEPRECATED_REASON("Y2023: Use nodeList() instead.")
  ItemInternalVectorView internalNodes() const { return _connectivity()->nodesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeList() instead.")
  ItemInternalVectorView internalEdges() const { return _connectivity()->edgesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceList() instead.")
  ItemInternalVectorView internalFaces() const { return _connectivity()->facesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellList() instead.")
  ItemInternalVectorView internalCells() const { return _connectivity()->cellsV2(m_local_id); }
  //@}

  /*!
   * \brief Méthodes utilisant les nouvelles connectivités pour accéder
   * aux informations de connectivité. A ne pas utiliser en dehors de Arcane.
   *
   * \warning Ces méthodes ne doivent être appelées que sur les entités
   * qui possèdent la connectivité associée.
   * Par exemple, cela ne fonctionne pas sur Cell->Cell car il n'y a pas de
   * connectivité maille/maille. En cas de mauvaise utilisation, cela
   * se traduit par un débordement de tableau.
   */
  //@{
  impl::ItemIndexedListView<DynExtent> nodeList() const { return _connectivity()->nodeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> edgeList() const { return _connectivity()->edgeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> faceList() const { return _connectivity()->faceList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> cellList() const { return _connectivity()->cellList(m_local_id); }

  impl::ItemIndexedListView<DynExtent> itemList(Node*) const { return nodeList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Edge*) const { return edgeList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Face*) const { return faceList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Cell*) const { return cellList(); }
  //@}

  ItemBase nodeBase(Int32 index) const { return _connectivity()->nodeBase(m_local_id,index); }
  ItemBase edgeBase(Int32 index) const { return _connectivity()->edgeBase(m_local_id,index); }
  ItemBase faceBase(Int32 index) const { return _connectivity()->faceBase(m_local_id,index); }
  ItemBase cellBase(Int32 index) const { return _connectivity()->cellBase(m_local_id,index); }
  ItemBase hParentBase(Int32 index) const { return _connectivity()->hParentBase(m_local_id, index, m_shared_info); }
  ItemBase hChildBase(Int32 index) const { return _connectivity()->hChildBase(m_local_id, index, m_shared_info); }
  inline ItemBase parentBase(Int32 index) const;

 public:

 /*!
   * @returns le rang de l'enfant \p (iitem).
   * exemple: si rank = m_internal->whichChildAmI(iitem); donc
   * m_internal->hChild(rank) serait iitem;
   */
  Int32 whichChildAmI(Int32 local_id) const;

 public:

  ItemBase topHParentBase() const;

 public:

  //! Interface modifiable de cette entité
  inline MutableItemBase toMutable();

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane.")
  inline ItemInternal* itemInternal() const;

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane.")
  ItemInternalVectorView _internalActiveCells(Int32Array& local_ids) const
  {
    return _internalActiveCells2(local_ids);
  }

 private:

  /*!
   * \brief Numéro local (au sous-domaine) de l'entité.
   *
   * Pour des raisons de performance, le numéro local doit être
   * le premier champs de la classe.
   */
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;

  //! Champ servant uniquement à gérer explicitement l'alignement
  Int32 m_padding = 0;

  //! Infos partagées entre toutes les entités ayant les mêmes caractéristiques
  ItemSharedInfo* m_shared_info = nullptr;

 private:

  ItemInternalConnectivityList* _connectivity() const
  {
    return m_shared_info->m_connectivity;
  }
  void _setFromInternal(ItemBase* rhs)
  {
    m_local_id = rhs->m_local_id;
    m_shared_info = rhs->m_shared_info;
  }
  void _setFromInternal(const ItemBase& rhs)
  {
    m_local_id = rhs.m_local_id;
    m_shared_info = rhs.m_shared_info;
  }
  ItemInternalVectorView _internalActiveCells2(Int32Array& local_ids) const;
  inline ItemInternal* _itemInternal() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Méthodes permettant de modifier ItemBase.
 *
 * Ces méthodes sont internes à Arcane.
 */
class ARCANE_CORE_EXPORT MutableItemBase
: public ItemBase
{
  friend class ::Arcane::Item;
  friend ItemBase;
  // Pour _setFaceBackAndFrontCell()
  friend Arcane::mesh::FaceFamily;

 private:

  MutableItemBase(Int32 local_id,ItemSharedInfo* shared_info)
  : ItemBase(local_id, shared_info) {}

 public:

  MutableItemBase() = default;
  MutableItemBase(ItemBaseBuildInfo x) : ItemBase(x) {}
  explicit MutableItemBase(const ItemBase& x)
  : ItemBase(x)
  {}

 public:

  // TODO: A supprimer à terme
  inline MutableItemBase(ItemInternal* x);

 public:

  void setUniqueId(Int64 uid)
  {
    _checkUniqueId(uid);
    m_shared_info->m_unique_ids[m_local_id] = uid;
  }

  //! Annule l'uniqueId a la valeur NULL_ITEM_UNIQUE_ID
  /*! Controle que la valeur à annuler est valid en mode ARCANE_CHECK */
  void unsetUniqueId();

  /*!
   * \brief Positionne le numéro du sous-domaine propriétaire de l'entité.

    \a current_sub_domain est le numéro du sous-domaine appelant cette opération.

    Après appel à cette fonction, il faut mettre à jour le maillage auquel cette entité
    appartient en appelant la méthode IMesh::notifyOwnItemsChanged(). Il n'est pas
    nécessaire de faire appel à cette méthode pour chaque appel de setOwn. Un seul
    appel après l'ensemble des modification est nécessaire.
  */
  void setOwner(Integer suid,Int32 current_sub_domain)
  {
    m_shared_info->_setOwnerV2(m_local_id,suid);
    int f = flags();
    if (suid==current_sub_domain)
      f |= II_Own;
    else
      f &= ~II_Own;
    setFlags(f);
  }

  //! Positionne les flags de l'entité
  void setFlags(Int32 f) { m_shared_info->_setFlagsV2(m_local_id,f); }

  //! Ajoute les flags \added_flags à ceux de l'entité
  void addFlags(Int32 added_flags)
  {
    Int32 f = this->flags();
    f |= added_flags;
    this->setFlags(f);
  }

  //! Supprime les flags \added_flags de ceux de l'entité
  void removeFlags(Int32 removed_flags)
  {
    Int32 f = this->flags();
    f &= ~removed_flags;
    this->setFlags(f);
  }

  //! Positionne l'état détachée de l'entité
  void setDetached(bool v)
  {
    int f = flags();
    if (v)
      f |= II_Detached;
    else
      f &= ~II_Detached;
    setFlags(f);
  }

  void reinitialize(Int64 uid,Int32 aowner,Int32 owner_rank)
  {
    setUniqueId(uid);
    setFlags(0);
    setOwner(aowner,owner_rank);
  }

  void setLocalId(Int32 local_id)
  {
    m_local_id = local_id;
  }
 
  //! Positionne le \a i-ème parent (actuellement aindex doit valoir 0)
  void setParent(Int32 aindex,Int32 parent_local_id)
  {
    m_shared_info->_setParentV2(m_local_id,aindex,parent_local_id);
  }

 private:

  void _setFaceBackAndFrontCells(Int32 back_cell_lid, Int32 front_cell_lid);

  void _checkUniqueId(Int64 new_uid) const;

  inline void _setFaceInfos(Int32 mod_flags);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Structure interne d'une entité de maillage.

 Cette instance contient la structure interne d'une entité de maillage.
 Elle ne doit être manipulée que par ceux qui savent ce qu'il font...

 Pour utiliser une entité, il faut utiliser la classe Item ou l'une
 de ces classes dérivées.

 En règle général, le maillage (IMesh) auquel l'entité appartient maintient
 différentes structures permettant de manipuler le maillage. Ces structures
 sont souvent recalculés dynamiquement lorsque cela est nécessaire (lazy
 evaluation). C'est le cas par exemple des groupes d'entités propres
 au sous-domaine ou de la table de conversion des numéros globaux en
 numéros locaux. C'est pourquoi il est primordial lorqu'on effectue
 une série de modifications d'instances de cette classe de notifier
 le maillage des changements effectués.
 */
class ARCANE_CORE_EXPORT ItemInternal
: public impl::MutableItemBase
{
  // Pour accès à _setSharedInfo()
  friend class mesh::DynamicMeshKindInfos;
  friend class mesh::ItemFamily;

 public:

  //! Entité nulle
  static ItemInternal nullItemInternal;
  static ItemInternal* nullItem() { return &nullItemInternal; }

 public:

  // Il faut utiliser la méthode correspondante de ItemBase

  //! Maille connectée à l'entité si l'entité est une entité sur la frontière (0 si aucune)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::boundaryCell() instead.")
  ItemInternal* boundaryCell() const { return (flags() & II_Boundary) ? _internalCell(0) : nullItem(); }
  //! Maille derrière l'entité (nullItem() si aucune)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::backCell() instead.")
  ItemInternal* backCell() const
  {
    if (flags() & II_HasBackCell)
      return _internalCell((flags() & II_BackCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  //! Maille devant l'entité (nullItem() si aucune)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::frontCell() instead.")
  ItemInternal* frontCell() const
  {
    if (flags() & II_HasFrontCell)
      return _internalCell((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::masterFace() instead.")
  ItemInternal* masterFace() const
  {
    if (flags() & II_SlaveFace)
      return _internalFace(0);
    return nullItem();
  }

 public:

  //! Infos partagées de l'entité.
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  ItemSharedInfo* sharedInfo() const { return m_shared_info; }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Node*) const { return nodeList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Edge*) const { return edgeList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Face*) const { return faceList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Cell*) const { return cellList(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use nodeBase() instead.")
  ItemInternal* internalNode(Int32 index) const { return _connectivity()->_nodeV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeBase() instead.")
  ItemInternal* internalEdge(Int32 index) const { return _connectivity()->_edgeV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceBase() instead.")
  ItemInternal* internalFace(Int32 index) const { return _connectivity()->_faceV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellBase() instead.")
  ItemInternal* internalCell(Int32 index) const { return _connectivity()->_cellV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use hParentBase() instead.")
  ItemInternal* internalHParent(Int32 index) const { return _connectivity()->_hParentV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use hChildBase() instead.")
  ItemInternal* internalHChild(Int32 index) const { return _connectivity()->_hChildV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use parentBase() instead.")
  ItemInternal* parent(Integer index) const { return m_shared_info->_parentV2(m_local_id,index); }

 public:

  const ItemInternal* topHParent() const;
  ItemInternal* topHParent();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always returns 0")
  Int32 dataIndex() { return 0; }

 public:

  /*!
   * \brief Pointeur sur la liste des parents.
   *
   * Comme actuellement on ne supporte qu'un seul niveau il est uniquement autorisé
   * de faire parentPtr()[0]. Cela ne permet aucune vérification et il est
   * donc préférable d'utiliser parentId() ou setParent() à la place.
   *
   * Au mois de juillet 2022 cette méthode n'est plus utilisée dans Arcane donc si
   * aucun code ne l'utilise (ce qui devrait être le cas car il s'agit d'une méthode
   * interne) on pourra la supprimer rapidement.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use parentId() or setParent() instead")
  Int32* parentPtr() { return m_shared_info->_parentPtr(m_local_id); }

  /*!
   * @returns le rang de l'enfant \p (iitem).
   * exemple: si rank = m_internal->whichChildAmI(iitem); donc
   * m_internal->hChild(rank) serait iitem;
   */
  Int32 whichChildAmI(const ItemInternal *iitem) const;

 public:

  //! Mémoire nécessaire pour stocker les infos de l'entité
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Integer neededMemory() const { return 0; }

  //! Mémoire minimale nécessaire pour stocker les infos de l'entité (sans tampon)
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Integer minimumNeededMemory() const { return 0; }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use nodeId() instead")
  Int32 nodeLocalId(Integer index) { return _connectivity()->_nodeLocalIdV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeId() instead")
  Int32 edgeLocalId(Integer index) { return _connectivity()->_edgeLocalIdV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceId() instead")
  Int32 faceLocalId(Integer index) { return _connectivity()->_faceLocalIdV2(m_local_id,index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellId() instead")
  Int32 cellLocalId(Integer index) { return _connectivity()->_cellLocalIdV2(m_local_id,index); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setDataIndex(Integer);

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  void setSharedInfo(ItemSharedInfo* shared_infos,ItemTypeId type_id)
  {
    _setSharedInfo(shared_infos,type_id);
  }

 public:

  //! \internal
  typedef ItemInternal* ItemInternalPtr;

  //! \internal
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  static ItemSharedInfo* _getSharedInfo(const ItemInternalPtr* items)
  {
    return ((items) ? items[0]->m_shared_info : ItemSharedInfo::nullInstance());
  }

 private:

  void _setSharedInfo(ItemSharedInfo* shared_infos,ItemTypeId type_id)
  {
    m_shared_info = shared_infos;
    shared_infos->_setTypeId(m_local_id,type_id.typeId());
  }

  ItemInternal* _internalFace(Int32 index) const { return _connectivity()->_faceV2(m_local_id, index); }
  ItemInternal* _internalCell(Int32 index) const { return _connectivity()->_cellV2(m_local_id, index); }
  ItemInternal* _internalHParent(Int32 index) const { return _connectivity()->_hParentV2(m_local_id, index); }
  ItemInternal* _internalHChild(Int32 index) const { return _connectivity()->_hChildV2(m_local_id, index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemBase::
ItemBase(ItemInternal* x)
: m_local_id(x->m_local_id)
, m_shared_info(x->m_shared_info)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableItemBase::
MutableItemBase(ItemInternal* x)
: ItemBase(x)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(ItemInternal* item)
: m_local_id(item->localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: ajouter vérification du bon type
template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemInternal* item)
: ItemLocalId(item->localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* impl::ItemBase::
itemInternal() const
{
  if (m_local_id!=NULL_ITEM_LOCAL_ID)
    return m_shared_info->m_items_internal[m_local_id];
  return ItemInternal::nullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* impl::ItemBase::
_itemInternal() const
{
  if (m_local_id!=NULL_ITEM_LOCAL_ID)
    return m_shared_info->m_items_internal[m_local_id];
  return ItemInternal::nullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline impl::ItemBase impl::ItemBase::
parentBase(Int32 index) const
{
  return ItemBase(m_shared_info->_parentV2(m_local_id,index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline impl::MutableItemBase impl::ItemBase::
toMutable()
{
  return MutableItemBase(m_local_id,m_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId impl::ItemBase::
itemLocalId() const
{
  return ItemLocalId(m_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Méthodes pour conversions entre différentes classes de gestion
 * des entités
 *
 * Cette classe est temporaire et interne à Arcane. Seules les classes 'friend'
 * peuvent l'utiliser.
 */
class ItemInternalCompatibility
{
  friend class SimdItemBase;
  friend class SimdItemDirectBase;
	friend class SimdItem;
  friend class SimdItemEnumeratorBase;
  friend class ItemVectorView;
  template<typename T> friend class ItemEnumeratorBaseT;
  friend class mesh::DynamicMeshKindInfos;
  friend class TotalviewAdapter;
  template<int Extent> friend class ItemConnectedListView;

 private:

  //! \internal
  typedef ItemInternal* ItemInternalPtr;
  static ItemSharedInfo* _getSharedInfo(const ItemInternal* item)
  {
    return item->m_shared_info;
  }
  static ItemSharedInfo* _getSharedInfo(const ItemInternalPtr* items,Int32 count)
  {
    return ((items && count>0) ? items[0]->m_shared_info : ItemSharedInfo::nullInstance());
  }
  static const ItemInternalPtr* _getItemInternalPtr(ItemSharedInfo* shared_info)
  {
    ARCANE_CHECK_PTR(shared_info);
    return shared_info->m_items_internal.data();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
