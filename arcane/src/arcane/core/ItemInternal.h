﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.h                                              (C) 2000-2023 */
/*                                                                           */
/* Partie interne d'une entité.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINTERNAL_H
#define ARCANE_ITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ItemTypes.h"
#include "arcane/ItemIndexedListView.h"
#include "arcane/ItemSharedInfo.h"
#include "arcane/ItemUniqueId.h"
#include "arcane/ItemLocalId.h"
#include "arcane/ItemTypeId.h"
#include "arcane/ItemConnectivityContainerView.h"
#include "arcane/ItemInternalVectorView.h"

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
 * \brief Flags pour les caractéristiques des entités.
 */
class ARCANE_CORE_EXPORT ItemFlags
{
 public:
  enum
  { // L'affichage 'lisible' des flags est implémenté dans ItemPrinter
    // Il doit être updaté si des flags sont ici changés
    II_Boundary = 1 << 1, //!< L'entité est sur la frontière
    II_HasFrontCell = 1 << 2, //!< L'entité a une maille devant
    II_HasBackCell  = 1 << 3, //!< L'entité a une maille derrière
    II_FrontCellIsFirst = 1 << 4, //!< La première maille de l'entité est la maille devant
    II_BackCellIsFirst  = 1 << 5, //!< La première maille de l'entité est la maille derrière
    II_Own = 1 << 6, //!< L'entité est une entité propre au sous-domaine
    II_Added = 1 << 7, //!< L'entité vient d'être ajoutée
    II_Suppressed = 1 << 8, //!< L'entité vient d'être supprimée
    II_Shared = 1 << 9, //!< L'entité est partagée par un autre sous-domaine
    II_SubDomainBoundary = 1 << 10, //!< L'entité est à la frontière de deux sous-domaines
    //II_JustRemoved = 1 << 11, //!< L'entité vient d'être supprimé
    II_JustAdded = 1 << 12, //!< L'entité vient d'être ajoutée
    II_NeedRemove = 1 << 13, //!< L'entité doit être supprimé
    II_SlaveFace = 1 << 14, //!< L'entité est une face esclave d'une interface
    II_MasterFace = 1 << 15, //!< L'entité est une face maître d'une interface
    II_Detached = 1 << 16, //!< L'entité est détachée du maillage
    II_HasTrace = 1 << 17, //!< L'entité est marquée pour trace (pour débug)

    II_Coarsen = 1 << 18, //!<  L'entité est marquée pour déraffinement
    II_DoNothing = 1 << 19, //!<  L'entité est bloquée
    II_Refine = 1 << 20, //!<  L'entité est marquée pour raffinement
    II_JustRefined = 1 << 21, //!<  L'entité vient d'être raffinée
    II_JustCoarsened = 1 << 22,//!<  L'entité vient d'être déraffiné
    II_Inactive = 1 << 23, //!<  L'entité est inactive //COARSEN_INACTIVE,
    II_CoarsenInactive = 1 << 24, //!<  L'entité est inactive et a des enfants tagués pour déraffinement

    II_UserMark1 = 1 << 25, //!< Marque utilisateur old_value 1<<24
    II_UserMark2 = 1 << 26 //!< Marque utilisateur  old_value 1<<25
  };
  static const int II_InterfaceFlags = II_Boundary + II_HasFrontCell + II_HasBackCell +
  II_FrontCellIsFirst + II_BackCellIsFirst;
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
      m_nb_item_null_data[i][0] = 0;
      m_nb_item_null_data[i][1] = 0;
      m_nb_item[i].setNull(&m_nb_item_null_data[i][1]);
      m_max_nb_item[i] = 0;
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

 public:

  /*!
   * \brief Liste des localId() des entités de type \a item_kind
   * connectées à l'entité de de localid() \a lid.
   */
  const Int32* itemLocalIds(Int32 item_kind,Int32 lid) const
  {
    const ItemLocalId* ptr =  &(m_list[item_kind][ m_indexes[item_kind][lid] ]);
    return reinterpret_cast<const Int32*>(ptr);
  }

  /*!
   * \brief localId() de la \a index-ème entité de type \a item_kind
   * connectés à l'entité de de localid() \a lid.
   */
  Int32 itemLocalId(Int32 item_kind,Int32 lid,Integer index) const
  {
    return m_list[item_kind][ m_indexes[item_kind][lid] + index];
  }
  //! Nombre d'appel à itemLocalId()
  Int64 nbAccess() const { return 0; }
  //! Nombre d'appel à itemLocalIds()
  Int64 nbAccessAll() const { return 0; }

 public:

  //! Positionne le tableau d'index des connectivités
  void setConnectivityIndex(Int32 item_kind,ConstArrayView<Int32> v)
  {
    m_indexes[item_kind] = v;
  }
  //! Positionne le tableau contenant la liste des connectivités
  void setConnectivityList(Int32 item_kind,ConstArrayView<Int32> v)
  {
    auto* ids = reinterpret_cast<const ItemLocalId*>(v.data());
    m_list[item_kind] = ConstArrayView<ItemLocalId>(v.size(),ids);
  }
  //! Positionne le tableau contenant la liste des connectivités
  void setConnectivityList(Int32 item_kind,ConstArrayView<ItemLocalId> v)
  {
    m_list[item_kind] = v;
  }
  //! Positionne le tableau contenant le nombre d'entités connectées.
  void setConnectivityNbItem(Int32 item_kind,ConstArrayView<Int32> v)
  {
    m_nb_item[item_kind] = v;
  }

 public:

  //! Positionne le nombre maximum d'entités connectées.
  void setMaxNbConnectedItem(Int32 item_kind,Int32 v)
  {
    m_max_nb_item[item_kind] = v;
  }

 public:

  //! Tableau d'index des connectivités pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityIndex(Int32 item_kind) const
  {
    return m_indexes[item_kind];
  }
  //! Tableau contenant la liste des connectivités pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityList(Int32 item_kind) const
  {
    auto* ids = reinterpret_cast<const Int32*>(m_list[item_kind].data());
    return { m_list[item_kind].size(), ids };
  }
  //! Tableau contenant le nombre d'entités connectées pour les entités de genre \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityNbItem(Int32 item_kind) const
  {
    return m_nb_item[item_kind];
  }

 public:

  //! Nombre maximum d'entités connectées.
  Int32 maxNbConnectedItem(Int32 item_kind) const
  {
    return m_max_nb_item[item_kind];
  }

  ItemConnectivityContainerView containerView(Int32 item_kind) const
  {
    return { m_list[item_kind], m_indexes[item_kind], m_nb_item[item_kind] };
  }

 public:

  ItemInternal* nodeV2(Int32 lid,Int32 aindex) const
  { return m_items->nodes[ _nodeLocalIdV2(lid,aindex) ]; }
  ItemInternal* edgeV2(Int32 lid,Int32 aindex) const
  { return m_items->edges[ _edgeLocalIdV2(lid,aindex) ]; }
  ItemInternal* faceV2(Int32 lid,Int32 aindex) const
  { return m_items->faces[ _faceLocalIdV2(lid,aindex) ]; }
  ItemInternal* cellV2(Int32 lid,Int32 aindex) const
  { return m_items->cells[ _cellLocalIdV2(lid,aindex) ]; }
  ItemInternal* hParentV2(Int32 lid,Int32 aindex) const
  { return m_items->cells[ _hParentLocalIdV2(lid,aindex) ]; }
  ItemInternal* hChildV2(Int32 lid,Int32 aindex) const
  { return m_items->cells[ _hChildLocalIdV2(lid,aindex) ]; }

  ItemBaseBuildInfo nodeBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_nodeLocalIdV2(lid,aindex),A_INTERNAL_SI(node)); }
  ItemBaseBuildInfo edgeBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_edgeLocalIdV2(lid,aindex),A_INTERNAL_SI(edge)); }
  ItemBaseBuildInfo faceBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_faceLocalIdV2(lid,aindex),A_INTERNAL_SI(face)); }
  ItemBaseBuildInfo cellBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_cellLocalIdV2(lid,aindex),A_INTERNAL_SI(cell)); }
  ItemBaseBuildInfo hParentBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_hParentLocalIdV2(lid,aindex),A_INTERNAL_SI(cell)); }
  ItemBaseBuildInfo hChildBase(Int32 lid,Int32 aindex) const
  { return ItemBaseBuildInfo(_hChildLocalIdV2(lid,aindex),A_INTERNAL_SI(cell)); }

  ItemInternalVectorView nodesV2(Int32 lid) const
  { return ItemInternalVectorView(A_INTERNAL_SI(node),_nodeLocalIdsV2(lid),_nbNodeV2(lid)); }
  ItemInternalVectorView edgesV2(Int32 lid) const
  { return ItemInternalVectorView(A_INTERNAL_SI(edge),_edgeLocalIdsV2(lid),_nbEdgeV2(lid)); }
  ItemInternalVectorView facesV2(Int32 lid) const
  { return ItemInternalVectorView(A_INTERNAL_SI(face),_faceLocalIdsV2(lid),_nbFaceV2(lid)); }
  ItemInternalVectorView cellsV2(Int32 lid) const
  { return ItemInternalVectorView(A_INTERNAL_SI(cell),_cellLocalIdsV2(lid),_nbCellV2(lid)); }

  auto nodeList(Int32 lid) const
  { return impl::ItemIndexedListView(A_INTERNAL_SI(node),_nodeLocalIdsV2(lid),_nbNodeV2(lid)); }
  auto edgeList(Int32 lid) const
  { return impl::ItemIndexedListView(A_INTERNAL_SI(edge),_edgeLocalIdsV2(lid),_nbEdgeV2(lid)); }
  auto faceList(Int32 lid) const
  { return impl::ItemIndexedListView(A_INTERNAL_SI(face),_faceLocalIdsV2(lid),_nbFaceV2(lid)); }
  auto cellList(Int32 lid) const
  { return impl::ItemIndexedListView(A_INTERNAL_SI(cell),_cellLocalIdsV2(lid),_nbCellV2(lid)); }

  Int32ConstArrayView nodeLocalIdsV2(Int32 lid) const
  { return Int32ConstArrayView(_nbNodeV2(lid),_nodeLocalIdsV2(lid)); }
  Int32ConstArrayView edgeLocalIdsV2(Int32 lid) const
  { return Int32ConstArrayView(_nbEdgeV2(lid), _edgeLocalIdsV2(lid)); }
  Int32ConstArrayView faceLocalIdsV2(Int32 lid) const
  { return Int32ConstArrayView(_nbFaceV2(lid), _faceLocalIdsV2(lid)); }
  Int32ConstArrayView cellLocalIdsV2(Int32 lid) const
  { return Int32ConstArrayView(_nbCellV2(lid),_cellLocalIdsV2(lid)); }

 public:

  const Int32* _nodeLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::NODE_IDX,lid); }
  const Int32* _edgeLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::EDGE_IDX,lid); }
  const Int32* _faceLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::FACE_IDX,lid); }
  const Int32* _cellLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::CELL_IDX,lid); }
  const Int32* _hParentLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::HPARENT_IDX,lid); }
  const Int32* _hChildLocalIdsV2(Int32 lid) const
  { return itemLocalIds(ItemInternalConnectivityList::HCHILD_IDX,lid); }
  Int32 _nodeLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::NODE_IDX,lid,index); }
  Int32 _edgeLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::EDGE_IDX,lid,index); }
  Int32 _faceLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::FACE_IDX,lid,index); }
  Int32 _cellLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::CELL_IDX,lid,index); }
  Int32 _hParentLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::HPARENT_IDX,lid,index); }
  Int32 _hChildLocalIdV2(Int32 lid,Int32 index) const
  { return itemLocalId(ItemInternalConnectivityList::HCHILD_IDX,lid,index); }

 public:

  Int32 _nbNodeV2(Int32 lid) const { return m_nb_item[NODE_IDX][lid]; }
  Int32 _nbEdgeV2(Int32 lid) const { return m_nb_item[EDGE_IDX][lid]; }
  Int32 _nbFaceV2(Int32 lid) const { return m_nb_item[FACE_IDX][lid]; }
  Int32 _nbCellV2(Int32 lid) const { return m_nb_item[CELL_IDX][lid]; }
  Int32 _nbHParentV2(Int32 lid) const { return m_nb_item[HPARENT_IDX][lid]; }
  Int32 _nbHChildrenV2(Int32 lid) const { return m_nb_item[HCHILD_IDX][lid]; }

 private:

  ConstArrayView<Int32> m_indexes[MAX_ITEM_KIND];
  Int32View m_nb_item[MAX_ITEM_KIND];
  ConstArrayView<ItemLocalId> m_list[MAX_ITEM_KIND];
  Int32 m_max_nb_item[MAX_ITEM_KIND];

 public:

  MeshItemInternalList* m_items;

 private:

#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
 impl::MeshItemSharedInfoList m_shared_infos;
#endif

 private:

  Int32 m_nb_item_null_data[MAX_ITEM_KIND][2];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour les entités du maillage.
 *
 * Cette classe est interne à %Arcane.
 *
 * Cette classe est la classe de base commune à Item et ItemInternal.
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
  friend MutableItemBase;

 private:

  ItemBase(Int32 local_id,ItemSharedInfo* shared_info)
  : m_local_id(local_id), m_shared_info(shared_info) {}

 public:

  ItemBase() : m_shared_info(ItemSharedInfo::nullItemSharedInfoPointer) {}
  ItemBase(ItemBase* x) : m_local_id(x->m_local_id), m_shared_info(x->m_shared_info) {}
  ItemBase(ItemBaseBuildInfo x) : m_local_id(x.m_local_id), m_shared_info(x.m_shared_info) {}

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
  bool isOwn() const { return (flags() & II_Own)!=0; }
  /*!
   * \brief Vrai si l'entité est partagé d'autres sous-domaines.
   *
   * Cette méthode n'est pertinente que si les informations de connectivités
   * ont été calculées.
   */
  bool isShared() const { return (flags() & II_Shared)!=0; }

  //! Vrai si l'entité est supprimée
  bool isSuppressed() const { return (flags() & II_Suppressed)!=0; }
  //! Vrai si l'entité est détachée
  bool isDetached() const { return (flags() & II_Detached)!=0; }

  //! \a true si l'entité est sur la frontière
  bool isBoundary() const { return (flags() & II_Boundary)!=0; }
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
  // TODO rendre obsolète après la version 3.8 (utiliser nodeList() à la place)
  ItemInternalVectorView internalNodes() const { return _connectivity()->nodesV2(m_local_id); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalEdges() const { return _connectivity()->edgesV2(m_local_id); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalFaces() const { return _connectivity()->facesV2(m_local_id); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalCells() const { return _connectivity()->cellsV2(m_local_id); }

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
  ItemIndexedListView<DynExtent> nodeList() const { return _connectivity()->nodeList(m_local_id); }
  ItemIndexedListView<DynExtent> edgeList() const { return _connectivity()->edgeList(m_local_id); }
  ItemIndexedListView<DynExtent> faceList() const { return _connectivity()->faceList(m_local_id); }
  ItemIndexedListView<DynExtent> cellList() const { return _connectivity()->cellList(m_local_id); }

  ItemIndexedListView<DynExtent> itemList(Node*) const { return nodeList(); }
  ItemIndexedListView<DynExtent> itemList(Edge*) const { return edgeList(); }
  ItemIndexedListView<DynExtent> itemList(Face*) const { return faceList(); }
  ItemIndexedListView<DynExtent> itemList(Cell*) const { return cellList(); }

  Int32ConstArrayView nodeIds() const { return _connectivity()->nodeLocalIdsV2(m_local_id); }
  Int32ConstArrayView edgeIds() const { return _connectivity()->edgeLocalIdsV2(m_local_id); }
  Int32ConstArrayView faceIds() const { return _connectivity()->faceLocalIdsV2(m_local_id); }
  Int32ConstArrayView cellIds() const { return _connectivity()->cellLocalIdsV2(m_local_id); }
  //@}

  ItemBase nodeBase(Int32 index) const { return _connectivity()->nodeBase(m_local_id,index); }
  ItemBase edgeBase(Int32 index) const { return _connectivity()->edgeBase(m_local_id,index); }
  ItemBase faceBase(Int32 index) const { return _connectivity()->faceBase(m_local_id,index); }
  ItemBase cellBase(Int32 index) const { return _connectivity()->cellBase(m_local_id,index); }
  ItemBase hParentBase(Int32 index) const { return _connectivity()->hParentBase(m_local_id,index); }
  ItemBase hChildBase(Int32 index) const { return _connectivity()->hChildBase(m_local_id,index); }
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

  // TODO: rendre obsolète
  inline ItemInternal* itemInternal() const;

  // TODO rendre obsolète
  ItemInternalVectorView _internalActiveCells(Int32Array& local_ids) const;

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

 private:

  MutableItemBase(Int32 local_id,ItemSharedInfo* shared_info)
  : ItemBase(local_id, shared_info) {}

 public:

  MutableItemBase() = default;
  MutableItemBase(ItemBase* x) : ItemBase(x) {}
  MutableItemBase(ItemBaseBuildInfo x) : ItemBase(x) {}

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

 public:

  /*!
   * \brief Méthodes temporaires pour passer des connectivités historiques
   * aux nouvelles. A ne pas utiliser en dehors de Arcane.
   */
  //@{
  /*!
   * \internal
   * \brief Pour une face, positionne à la fois la back cell et la front cell.
   *
   * \a back_cell_lid et/ou \a front_cell_lid peuvent valoir NULL_ITEM_LOCAL_ID
   * ce qui signifie que l'entité n'a pas de back cell ou front cell. Si les
   * deux valeurs sont nulles, alors la face est considérée comme n'ayant
   * plus de mailles connectées.
   */
  void _setFaceBackAndFrontCells(Int32 back_cell_lid,Int32 front_cell_lid);
  //@}

 private:

  void _checkUniqueId(Int64 new_uid) const;

  inline void _setFaceInfos(Int32 mod_flags);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

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

  //! Maille connectée à l'entité si l'entité est une entité sur la frontière (0 si aucune)
  ItemInternal* boundaryCell() const { return (flags() & II_Boundary) ? internalCell(0) : nullItem(); }
  //! Maille derrière l'entité (nullItem() si aucune)
  ItemInternal* backCell() const
  {
    if (flags() & II_HasBackCell)
      return internalCell((flags() & II_BackCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  //! Maille devant l'entité (nullItem() si aucune)
  ItemInternal* frontCell() const
  {
    if (flags() & II_HasFrontCell)
      return internalCell((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  ItemInternal* masterFace() const
  {
    if (flags() & II_SlaveFace)
      return internalFace(0);
    return nullItem();
  }

  //! Infos partagées de l'entité.
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  ItemSharedInfo* sharedInfo() const { return m_shared_info; }

 public:

  // TODO rendre obsolète après la version 3.8 (utiliser itemList() à la place)
  ItemInternalVectorView internalItems(Node*) const { return nodeList(); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalItems(Edge*) const { return edgeList(); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalItems(Face*) const { return faceList(); }
  // TODO rendre obsolète après la version 3.8
  ItemInternalVectorView internalItems(Cell*) const { return cellList(); }

 public:

  ItemInternal* internalNode(Int32 index) const { return _connectivity()->nodeV2(m_local_id,index); }
  ItemInternal* internalEdge(Int32 index) const { return _connectivity()->edgeV2(m_local_id,index); }
  ItemInternal* internalFace(Int32 index) const { return _connectivity()->faceV2(m_local_id,index); }
  ItemInternal* internalCell(Int32 index) const { return _connectivity()->cellV2(m_local_id,index); }
  ItemInternal* internalHParent(Int32 index) const { return _connectivity()->hParentV2(m_local_id,index); }
  ItemInternal* internalHChild(Int32 index) const { return _connectivity()->hChildV2(m_local_id,index); }
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

  // TODO: rendre obsolète après la version 3.8 les 4 méthodes suivantes
  // car il faut utiliser nodeId(), edgeId(), ...
  Int32 nodeLocalId(Integer index) { return _connectivity()->_nodeLocalIdV2(m_local_id,index); }
  Int32 edgeLocalId(Integer index) { return _connectivity()->_edgeLocalIdV2(m_local_id,index); }
  Int32 faceLocalId(Integer index) { return _connectivity()->_faceLocalIdV2(m_local_id,index); }
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
};

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
