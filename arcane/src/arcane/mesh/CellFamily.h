// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamily.h                                                (C) 2000-2022 */
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
  Item allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info) override;
  Item findOrAllocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily* family() override {return this;}

  Cell allocOne(Int64 uid,ItemTypeId type);
  Cell findOrAllocOne(Int64 uid,ItemTypeId type_id,bool& is_alloc);
  
  /*!
   * Supprime la maille \a cell
   *
   * @param cell la maille à supprimer
   */
  void removeCell(Cell cell);

  /*!
   * Detache la maille \a cell du maillage sans la supprimer
   *
   * @param cell la maille à détacher
   */
  void detachCell(Cell cell);

  /*!
   * Detache les mailles d'identifiants locaux \a cell_local_ids du maillage sans les supprimer.
   * Basé sur le graphe de dépendances des familles ItemFamilyNetwork.
   *
   * @param cells_local_id identifiants locaux des mailles à détacher
   */
  void detachCells2(Int32ConstArrayView cell_local_ids);

  /*!
   * Detruit la maille \a cell ayant deja ete detachée du maillage
   *
   * @param cell la maille détachée à detruire
   */
  void removeDetachedCell(Cell cell);

  /*!
   * Supprime le groupe d'entités \a local_ids
   *
   * @param local_ids le groupe de mailles à supprimer
   */
  virtual void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost=false) override;

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
  void _addParentCellToCell(Cell cell,Cell parent_cell);
  void _addChildCellToCell(Cell parent_cell,Integer rank,Cell child_cell);
  void _addChildCellToCell2(Cell parent_cell,Cell child_cell);
  void _addChildrenCellsToCell(Cell parent_cell,Int32ConstArrayView children_cells_lid);
  void _removeParentCellToCell(Cell cell);
  void _removeChildCellToCell(Cell parent_cell,Cell cell);
  void _removeChildrenCellsToCell(Cell parent_cell);

 public:

  virtual void computeSynchronizeInfos() override;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use allocOne(Int64 uid,ItemTypeId type) instead")
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type);

  ARCANE_DEPRECATED_REASON("Y2022: Use findOrAllocOne(Int64 uid,ItemTypeId type_id,bool& is_alloc) instead")
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,bool& is_alloc);

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
  inline void _createOne(ItemInternal* item,Int64 uid,ItemTypeId type_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
