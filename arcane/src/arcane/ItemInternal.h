﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.h                                              (C) 2000-2020 */
/*                                                                           */
/* Partie interne d'une entité.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINTERNAL_H
#define ARCANE_ITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/ItemSharedInfo.h"
#include "arcane/ItemUniqueId.h"
#include "arcane/ItemLocalId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Cette macro permet de savoir si on accède aux nouvelles connectivités
 * via une instance partagée dans ItemSharedInfo (si elle est définie) ou
 * si chaque entité à une copie de la connectivité. La première solution
 * fait une indirection supplémentaire et la seconde nécessite un pointeur
 * et un accès mémoire supplémentaire par entité. A priori la première
 * solution est préférable si cela ne pose pas de problèmes de performance.
 */
#define ARCANE_USE_SHAREDINFO_CONNECTIVITY

#if ARCANE_ITEM_CONNECTIVITY_SIZE_MODE!=ARCANE_ITEM_CONNECTIVITY_SIZE_MODE_NEW
#error "Invalid configuration for ARCANE_ITEM_CONNECTIVITY_SIZE_MODE"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef null
#undef null
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Déclare temporairement les classes gérant les anciens accesseurs qui ont
 * besoin d'avoir accès aux méthodes privées de 'ItemInternal'.
 */
namespace Arcane::mesh
{
class StandardItemFamilyCompactPolicy;
class ParticleFamilyCompactPolicy;
class NodeFamily;
}

namespace Arcane
{

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
    void operator=(Int32ArrayView v)
    {
      m_data = v.data();
      m_size = v.size();
    }
    // Il faut que data[NULL_ITEM_LOCAL_ID] soit valide.
    void setNull(Int32* data)
    {
      m_data = data;
      m_size = 0;
    }
    operator Int32ConstArrayView() const
    {
      return Int32ConstArrayView(m_size,m_data);
    }
    Int32ArrayView toMutableView() const
    {
      return Int32ArrayView(m_size,m_data);
    }
   private:
    Int32 m_size;
    Int32* m_data;
  };
 public:
  enum {
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
  : m_items(nullptr), m_nb_access_all(0), m_nb_access(0)
  {
    for( Integer i=0; i<MAX_ITEM_KIND; ++i ){
      m_indexes_array[i] = nullptr;
      m_nb_item_array[i] = nullptr;
      m_nb_item_null_data[i][0] = 0;
      m_nb_item_null_data[i][1] = 0;
      m_nb_item[i].setNull(&m_nb_item_null_data[i][1]);
    }
  }
 public:
  /*!
   * \brief Liste des localId() des entités de type \a item_kind
   * connectées à l'entité de de localid() \a lid.
   */
  const Int32* itemLocalIds(Int32 item_kind,Int32 lid) const
  {
#ifdef ARCANE_CHECK
    ++m_nb_access_all;
#endif
    return &(m_list[item_kind][ m_indexes[item_kind][lid] ]);
  }
  /*!
   * \brief localId() de la \a index-ème entité de type \a item_kind
   * connectés à l'entité de de localid() \a lid.
   */
  Int32 itemLocalId(Int32 item_kind,Int32 lid,Integer index) const
  {
#ifdef ARCANE_CHECK
    ++m_nb_access;
#endif
    return m_list[item_kind][ m_indexes[item_kind][lid] + index];
  }
  //! Nombre d'appel à itemLocalId()
  Int64 nbAccess() const { return m_nb_access; }
  //! Nombre d'appel à itemLocalIds()
  Int64 nbAccessAll() const { return m_nb_access_all; }
  //! Positionne le tableau d'index des connectivités
  void setConnectivityIndex(Int32 item_kind,Int32ArrayView v)
  {
    m_indexes[item_kind] = v;
  }
  //! Positionne le tableau contenant la liste des connectivités
  void setConnectivityList(Int32 item_kind,Int32ConstArrayView v)
  {
    m_list[item_kind] = v;
  }
  //! Positionne le tableau contenant le nombre d'entités connectées.
  void setConnectivityNbItem(Int32 item_kind,Int32ArrayView v)
  {
    m_nb_item[item_kind] = v;
  }
  //! Tableau d'index des connectivités pour les entités de genre \a item_kind
  Int32ConstArrayView connectivityIndex(Int32 item_kind) const
  {
    return m_indexes[item_kind];
  }
  //! Tableau contenant la liste des connectivités pour les entités de genre \a item_kind
  Int32ConstArrayView connectivityList(Int32 item_kind) const
  {
    return m_list[item_kind];
  }
  //! Tableau contenant le nombre d'entités connectées pour les entités de genre \a item_kind
  Int32ConstArrayView connectivityNbItem(Int32 item_kind) const
  {
    return m_nb_item[item_kind];
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
  ItemInternalVectorView nodesV2(Int32 lid) const
  { return ItemInternalVectorView(m_items->nodes,_nodeLocalIdsV2(lid),_nbNodeV2(lid)); }
  ItemInternalVectorView edgesV2(Int32 lid) const
  { return ItemInternalVectorView(m_items->edges,_edgeLocalIdsV2(lid),_nbEdgeV2(lid)); }
  ItemInternalVectorView facesV2(Int32 lid) const
  { return ItemInternalVectorView(m_items->faces,_faceLocalIdsV2(lid),_nbFaceV2(lid)); }
  ItemInternalVectorView cellsV2(Int32 lid) const
  { return ItemInternalVectorView(m_items->cells,_cellLocalIdsV2(lid),_nbCellV2(lid)); }

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
  // Ces deux tableaux utilisent une vue modifiable par compatibilité avec les anciens
  // mécanismes. Une fois que ceci auront été supprimés alors il sera
  // constant
  Int32ArrayView m_indexes[MAX_ITEM_KIND];
  Int32View m_nb_item[MAX_ITEM_KIND];
  Int32ConstArrayView m_list[MAX_ITEM_KIND];
 public:
  MeshItemInternalList* m_items;
 private:
  // Compte le nombre d'accès pour vérification. A supprimer par la suite.
  mutable Int64 m_nb_access_all;
  mutable Int64 m_nb_access;
 public:
  Int32ArrayView _mutableConnectivityIndex(Int32 item_kind) const
  {
    return m_indexes[item_kind];
  }
  Int32ArrayView _mutableConnectivityNbItem(Int32 item_kind) const
  {
    return m_nb_item[item_kind].toMutableView();
  }
  Int32Array* m_indexes_array[MAX_ITEM_KIND];
  Int32Array* m_nb_item_array[MAX_ITEM_KIND];
 private:
 Int32 m_nb_item_null_data[MAX_ITEM_KIND][2];
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
{
  friend class Arcane::mesh::StandardItemFamilyCompactPolicy;
  friend class Arcane::mesh::ParticleFamilyCompactPolicy;
  friend class Arcane::mesh::NodeFamily;
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

 public:
  //! Entité nulle
  static ItemInternal nullItemInternal;
  static ItemInternal* nullItem() { return &nullItemInternal; }
 public:
  ItemInternal() : m_local_id(NULL_ITEM_LOCAL_ID), m_data_index(0),
  m_shared_info(&ItemSharedInfo::nullItemSharedInfo)
#ifndef ARCANE_USE_SHAREDINFO_CONNECTIVITY
  , m_connectivity(&ItemInternalConnectivityList::nullInstance)
#endif
  {}
 public:
  //! Numéro local (au sous-domaine) de l'entité
  Integer localId() const { return m_local_id; }
  //! Numéro unique de l'entité
  ItemUniqueId uniqueId() const
  {
#ifdef ARCANE_CHECK
    if (m_local_id!=NULL_ITEM_LOCAL_ID)
      arcaneCheckAt((Integer)m_local_id,m_shared_info->m_unique_ids->size());
#endif
    // Ne pas utiliser l'accesseur normal car ce tableau peut etre utilise pour la maille
    // nulle et dans ce cas m_local_id vaut NULL_ITEM_LOCAL_ID (qui est negatif)
    // ce qui provoque une exception pour debordement de tableau.
    return ItemUniqueId(m_shared_info->m_unique_ids->data()[m_local_id]);
  }
  void setUniqueId(Int64 uid)
  {
    _checkUniqueId(uid);
    (*m_shared_info->m_unique_ids)[m_local_id] = uid;
  }

  //! Annule l'uniqueId a la valeur NULL_ITEM_UNIQUE_ID
  /*! Controle que la valeur à annuler est valid en mode ARCANE_CHECK */
  void unsetUniqueId();

  /*! \brief Positionne le numéro du sous-domaine propriétaire de l'entité.

    \a current_sub_domain est le numéro du sous-domaine appelant cette opération.

    Après appel à cette fonction, il faut mettre à jour le maillage auquel cette entité
    appartient en appelant la méthode IMesh::notifyOwnItemsChanged(). Il n'est pas
    nécessaire de faire appel à cette méthode pour chaque appel de setOwn. Un seul
    appel après l'ensemble des modification est nécessaire.
  */
  void setOwner(Integer suid,Int32 current_sub_domain)
  {
    m_shared_info->setOwner(m_data_index,suid);
    int f = flags();
    if (suid==current_sub_domain)
      f |= II_Own;
    else
      f &= ~II_Own;
    setFlags(f);
  }

  //! Numéro du sous-domaine propriétaire de l'entité
  Int32 owner() const { return m_shared_info->owner(m_data_index); }

  //! Flags de l'entité
  Int32 flags() const { return m_shared_info->flags(m_data_index); }

  //! Positionne les flags de l'entité
  void setFlags(Int32 f)  { m_shared_info->setFlags(m_data_index,f); }

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

  //! Nombre de noeuds de l'entité
  Integer nbNode() const { return _connectivity()->_nbNodeV2(m_local_id); }
  //! Nombre d'arêtes de l'entité ou nombre d'arêtes connectés à l'entités (pour les noeuds)
  Integer nbEdge() const { return _connectivity()->_nbEdgeV2(m_local_id); }
  //! Nombre de faces de l'entité ou nombre de faces connectés à l'entités (pour les noeuds et arêtes)
  Integer nbFace() const { return _connectivity()->_nbFaceV2(m_local_id); }
  //! Nombre de mailles connectées à l'entité (pour les noeuds, arêtes et faces)
  Integer nbCell() const { return _connectivity()->_nbCellV2(m_local_id); }
  //! Nombre de parent
  Int32 nbHParent() const { return _connectivity()->_nbHParentV2(m_local_id); }
  //! Nombre d' enfants
  Int32 nbHChildren() const { return _connectivity()->_nbHChildrenV2(m_local_id); }
  //! Nombre de parent
  Integer nbParent() const { return m_shared_info->nbParent(); }

  //! @returns \p true si l'item est actif (i.e. n'a pas de
  //! descendants actifs), \p false  sinon. Notez qu'il suffit de vérifier
  //! le premier enfant seulement. Renvoie toujours \p true si l'AMR est désactivé.
  inline bool isActive() const
  {
	  if ( (flags() & II_Inactive)  | (flags() & II_CoarsenInactive))
		  return false;
	  else
		  return true;
  }

  //! @returns le niveau de raffinement de l'item courant.  Si l'item
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
	  return (this->internalHParent(0)->level() + 1);
  }

  //! @returns \p true si l'item est un ancetre (i.e. a un
  //! enfant actif ou un enfant ancetre), \p false sinon. Renvoie toujours \p false si l'AMR est désactivé.
  inline bool isAncestor() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return false;
    if (this->internalHChild(0)->isActive())
      return true;
    return this->internalHChild(0)->isAncestor();
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
  //! @returns \p true si l'item est subactif (i.e. pas actif et n'a pas de
  //! descendants), \p false  sinon.Renvoie toujours \p false si l'AMR est désactivé.
  inline  bool isSubactive() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return true;
    return this->internalHChild(0)->isSubactive();
  }
  //! Numéro du type de l'entité
  Integer typeId() const { return m_shared_info->typeId(); }
  //! Type de l'entité.
  ItemTypeInfo* typeInfo() const { return m_shared_info->m_item_type; }
  //! Infos partagées de l'entité.
  ItemSharedInfo* sharedInfo() const { return m_shared_info; }
  //! Famille dont est issue l'entité
  IItemFamily* family() const { return m_shared_info->m_item_family; }
  //! Genre de l'entité
  eItemKind kind() const { return m_shared_info->m_item_kind; }
  //! Vrai si l'entité est l'entité nulle
  bool null() const { return m_local_id==NULL_ITEM_LOCAL_ID; }
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
  //! \a true si l'entité est sur la frontière
  bool isBoundary() const { return (flags() & II_Boundary)!=0; }
  //! Maille connectée à l'entité si l'entité est une entité sur la frontière (0 si aucune)
  ItemInternal* boundaryCell() const { return (flags() & II_Boundary) ? internalCell(0) : nullItem(); }
  //! Maille derrière l'entité (0 si aucune)
  ItemInternal* backCell() const
  {
    if (flags() & II_HasBackCell)
      return internalCell((flags() & II_BackCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  //! Maille devant l'entité (0 si aucune)
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
  //! \a true s'il s'agit de la face maître d'une interface
  inline bool isMasterFace() const { return flags() & II_MasterFace; }

  //! \a true s'il s'agit d'une face esclave d'une interface
  inline bool isSlaveFace() const { return flags() & II_SlaveFace; }

 public:
  void reinitialize(Int64 uid,Integer aowner,Int32 owner_rank)
  {
    setUniqueId(uid);
    setFlags(0);
    setOwner(aowner,owner_rank);
  }

 public:

  ItemInternalVectorView internalItems(Node*) const { return internalNodes(); }
  ItemInternalVectorView internalItems(Edge*) const { return internalEdges(); }
  ItemInternalVectorView internalItems(Face*) const { return internalFaces(); }
  ItemInternalVectorView internalItems(Cell*) const { return internalCells(); }

  //! AMR
  ItemInternalVectorView activeCells(Int32Array& local_ids) const;
  ItemInternalVectorView activeFaces(Int32Array& local_ids) const;
  ItemInternalVectorView activeEdges() const;

 public:

  ItemInternal* internalNode(Int32 index) const { return _connectivity()->nodeV2(m_local_id,index); }
  ItemInternal* internalEdge(Int32 index) const { return _connectivity()->edgeV2(m_local_id,index); }
  ItemInternal* internalFace(Int32 index) const { return _connectivity()->faceV2(m_local_id,index); }
  ItemInternal* internalCell(Int32 index) const { return _connectivity()->cellV2(m_local_id,index); }
  ItemInternal* internalHParent(Int32 index) const { return _connectivity()->hParentV2(m_local_id,index); }
  ItemInternal* internalHChild(Int32 index) const { return _connectivity()->hChildV2(m_local_id,index); }

 public:

  ItemInternal* parent(Integer index) const { return m_shared_info->parent(m_data_index,index); }

 public:

  const ItemInternal* topHParent() const;
  ItemInternal* topHParent();

 public:

  Int32 dataIndex() { return m_data_index; }

 private:

  Int32* dataPtr() { return m_shared_info->m_infos + m_data_index; }

 public:

  Int32* parentPtr() { return m_shared_info->m_infos + m_data_index + m_shared_info->firstParent(); }
  /**
   * @returns le rang de l'enfant \p (iitem).
   * exemple: si rank = m_internal->whichChildAmI(iitem); donc
   * m_internal->hChild(rank) serait iitem;
   */
  Int32 whichChildAmI(const ItemInternal *iitem) const;

 public:

  //! AMR
  void setParent(Integer aindex,Int32 local_id)
  {
    m_shared_info->setParent(m_data_index,aindex,local_id);
  }

  //! Mémoire nécessaire pour stocker les infos de l'entité
  Integer neededMemory() const { return m_shared_info->neededMemory(); }
  //! Mémoire minimale nécessaire pour stocker les infos de l'entité (sans tampon)
  Integer minimumNeededMemory() const { return m_shared_info->minimumNeededMemory(); }

 public:

  Int32 nodeLocalId(Integer index) { return _connectivity()->_nodeLocalIdV2(m_local_id,index); }
  Int32 edgeLocalId(Integer index) { return _connectivity()->_edgeLocalIdV2(m_local_id,index); }
  Int32 faceLocalId(Integer index) { return _connectivity()->_faceLocalIdV2(m_local_id,index); }
  Int32 cellLocalId(Integer index) { return _connectivity()->_cellLocalIdV2(m_local_id,index); }

#if 1
 private:

  Int32 _nodeLocalIdV2(Integer index) { return _connectivity()->_nodeLocalIdV2(m_local_id,index); }
  Int32 _edgeLocalIdV2(Integer index) { return _connectivity()->_edgeLocalIdV2(m_local_id,index); }
  Int32 _faceLocalIdV2(Integer index) { return _connectivity()->_faceLocalIdV2(m_local_id,index); }
  Int32 _cellLocalIdV2(Integer index) { return _connectivity()->_cellLocalIdV2(m_local_id,index); }
#endif

 public:

  void setLocalId(Int32 local_id) { m_local_id = local_id; }
  void setDataIndex(Integer index) { m_data_index = index; }
  void setSharedInfo(ItemSharedInfo* shared_infos)
  {
    m_shared_info = shared_infos;
#ifndef ARCANE_USE_SHAREDINFO_CONNECTIVITY
    m_connectivity = shared_infos->m_connectivity;
#endif
  }

 public:

  /*!
   * \brief Méthodes utilisant les nouvelles connectivités pour acceéder
   * aux informations de connectivité. A ne pas utiliser en dehors de Arcane.
   *
   * \warning Ces méthodes ne doivent être appelées que sur les entités
   * qui possèdent la connectivité associée ET qui sont nouveau format.
   * Par exemple, cela ne fonctionne pas sur Cell->Cell car il n'y a pas de
   * connectivité maille/maille, ni sur les Link car ils n'utilisent pas
   * les nouvelles connectivités. En cas de mauvaise utilisation, cela
   * se traduit par un débordement de tableau.
   * 
   */
  //@{
  ItemInternalVectorView internalNodes() const { return _connectivity()->nodesV2(m_local_id); }
  ItemInternalVectorView internalEdges() const { return _connectivity()->edgesV2(m_local_id); }
  ItemInternalVectorView internalFaces() const { return _connectivity()->facesV2(m_local_id); }
  ItemInternalVectorView internalCells() const { return _connectivity()->cellsV2(m_local_id); }

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

 public:
  void _internalCopyAndChangeSharedInfos(ItemSharedInfo* old_isi,ItemSharedInfo* new_isi,Integer new_data_index);
  void _internalCopyAndSetDataIndex(Int32* data_ptr,Int32 data_index);
  //@}

 private:
  /*!
   * \brief Numéro local (au sous-domaine) de l'entité.
   *
   * Pour des raisons de performance, le numéro local doit être
   * le premier champs de la classe.
   */
  Int32 m_local_id;
  //! Indice des données de cette entité dans le tableau des données.
  Int32 m_data_index;
  //!< Infos partagées entre toutes les entités ayant les mêmes caractéristiques
  ItemSharedInfo* m_shared_info;
#ifndef ARCANE_USE_SHAREDINFO_CONNECTIVITY
  //! Infos de connectivité nouvelle version (version 2017)
  ItemInternalConnectivityList* m_connectivity;
#endif

 private:

  void _checkUniqueId(Int64 new_uid) const;

  inline void _setFaceInfos(Int32 mod_flags);
  inline ItemInternalConnectivityList* _connectivity() const
  {
#ifdef ARCANE_USE_SHAREDINFO_CONNECTIVITY
    return m_shared_info->m_connectivity;
#else
    return m_connectivity;
#endif
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
