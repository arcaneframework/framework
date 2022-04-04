﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamily.h                                                (C) 2000-2021 */
/*                                                                           */
/* Famille de mailles.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CELLFAMILY_H
#define ARCANE_MESH_CELLFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemFamilyModifier.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeFamily;
class EdgeFamily;
class FaceFamily;
class CellFamily;
class HParentCellCompactIncrementalItemConnectivity;
class HChildCellCompactIncrementalItemConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille de mailles.
 */
class ARCANE_MESH_EXPORT CellFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class TopologyModifier;
  typedef ItemConnectivitySelectorT<NodeInternalConnectivityIndex,IncrementalItemConnectivity> NodeConnectivity;
  typedef ItemConnectivitySelectorT<EdgeInternalConnectivityIndex,IncrementalItemConnectivity> EdgeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex,IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<HParentInternalConnectivityIndex,IncrementalItemConnectivity> HParentConnectivity;
  typedef ItemConnectivitySelectorT<HChildInternalConnectivityIndex,IncrementalItemConnectivity> HChildConnectivity;
 public:

  CellFamily(IMesh* mesh,const String& name);
  virtual ~CellFamily(); //<! Libère les ressources

 public:

  virtual void build() override;
  virtual void preAllocate(Integer nb_item);

 public:

  // IItemFamilyModifier interface
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, MeshInfos& mesh_info) override;
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily*  family() override {return this;}

  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type);
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,bool& is_alloc);
  
  /**
   * Supprime la maille \a cell
   *
   * @param cell la maille à supprimer
   */
  void removeCell(ItemInternal* cell);

  /**
   * Detache la maille \a cell du maillage sans la supprimer
   *
   * @param cell la maille à détacher
   */
  void detachCell(ItemInternal* cell);

  /**
   * Detache les mailles d'identifiants locaux \a cell_local_ids du maillage sans les supprimer.
   * Basé sur le graphe de dépendances des familles ItemFamilyNetwork.
   *
   * @param cells_local_id identifiants locaux des mailles à détacher
   */
  void detachCells2(Int32ConstArrayView cell_local_ids);

  /**
   * Detruit la maille \a cell ayant deja ete detachée du maillage
   *
   * @param cell la maille détachée à detruire
   */
  void removeDetachedCell(ItemInternal* cell);

  /**
   * Supprime le groupe d'entités \a local_ids
   *
   * @param local_ids le groupe de mailles à supprimer
   */
  virtual void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost=false) override;

  /**
   * Fusionne deux entités en une plus grande. Par exemple, deux
   * mailles partageant une face. L'intérêt est de conserver ainsi les
   * uniqueIds() en parallèle.
   *
   * @note La maille résultante est construite en remplacement de la
   * première maille, de numéro \a local_id1
   *
   * @param local_id1 numéro local de la première entité
   * @param local_id2 numéro local de la seconde entité
   */
  void mergeItems(Int32 local_id1,Int32 local_id2) override;

  /**
   * Détermine quel sera le localId de la maille apres fusion.
   *
   * @param local_id1 numéro local de la première entité
   * @param local_id2 numéro local de la seconde entité
   *
   * @return le local id de la maille fusionnée
   */
  Int32 getMergedItemLID(Int32 local_id1,Int32 local_id2) override;

  //! Définit la connectivité active pour le maillage associé
  /*! Ceci conditionne les connectivités à la charge de cette famille */
  void setConnectivity(const Integer c);

 public:

  // TODO: rendre ces méthodes privées pour obliger à utiliser IItemFamilyTopologyModifier.
  void replaceNode(ItemLocalId cell,Integer index,ItemLocalId node);
  void replaceEdge(ItemLocalId cell,Integer index,ItemLocalId edge);
  void replaceFace(ItemLocalId cell,Integer index,ItemLocalId face);
  void replaceHChild(ItemLocalId cell,Integer index,ItemLocalId child_cell);
  void replaceHParent(ItemLocalId cell,Integer index,ItemLocalId parent_cell);

 public:

  //! AMR
  void _addParentCellToCell(ItemInternal* cell,ItemInternal* parent_cell);
  void _addChildCellToCell(ItemInternal* parent_cell,Integer rank,ItemInternal* child_cell);
  void _addChildrenCellsToCell(ItemInternal* parent_cell,Int32ConstArrayView children_cells_lid);
  void _removeParentCellToCell(ItemInternal* cell);
  void _removeChildCellToCell(ItemInternal* parent_cell,ItemInternal* cell);
  void _removeChildrenCellsToCell(ItemInternal* parent_cell);

 public:

  virtual void computeSynchronizeInfos() override;

 protected:

 private:

  Integer m_node_prealloc;
  Integer m_edge_prealloc;
  Integer m_face_prealloc;
  Integer m_mesh_connectivity;

  NodeFamily* m_node_family;
  EdgeFamily* m_edge_family;
  FaceFamily* m_face_family;

  NodeConnectivity* m_node_connectivity;
  EdgeConnectivity* m_edge_connectivity;
  FaceConnectivity* m_face_connectivity;
  HParentConnectivity* m_hparent_connectivity;
  HChildConnectivity* m_hchild_connectivity;

 private:

  /*! Assimilable à _removeOne dans les autres familles */
  void _removeSubItems(Cell cell);

  void _removeNotConnectedSubItems(Cell cell);
  inline void _createOne(ItemInternal* item,Int64 uid,ItemTypeInfo* type);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
