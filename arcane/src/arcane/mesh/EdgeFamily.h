// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeFamily.h                                                (C) 2000-2025 */
/*                                                                           */
/* Famille d'arêtes.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EDGEFAMILY_H
#define ARCANE_MESH_EDGEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemFamilyModifier.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille d'arêtes.
 */
class ARCANE_MESH_EXPORT EdgeFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class TopologyModifier;
  typedef ItemConnectivitySelectorT<NodeInternalConnectivityIndex,IncrementalItemConnectivity> NodeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex,IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex,IncrementalItemConnectivity> CellConnectivity;

 public:

  EdgeFamily(IMesh* mesh,const String& name);
  virtual ~EdgeFamily(); //<! Libère les ressources

 public:

  virtual void build() override;
  virtual void preAllocate(Integer nb_item);
  virtual void computeSynchronizeInfos() override;

 public:

  //! Version appelées dans l'ajout générique d'item
  // IItemFamilyModifier interface
  Item allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info) override;
  Item findOrAllocOne(Int64 uid,ItemTypeId type_id,MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily* family() override {return this;}

  ItemInternal* allocOne(Int64 uid);
  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc);

  void replaceNode(ItemLocalId edge,Integer index,ItemLocalId node);

  //! Ajoute une maille voisine à une arête
  void addCellToEdge(Edge edge,Cell new_cell);
  //! Ajoute une maille voisine à une arête
  void addFaceToEdge(Edge edge,Face new_face);
  //! Supprime une maille d'une arête
  void removeCellFromEdge(Edge edge,ItemLocalId cell_to_remove_lid);
  //! Supprime une maille d'une arête
  void removeFaceFromEdge(ItemLocalId edge,ItemLocalId face_to_remove);
  //! Supprime l'arête si elle n'est plus connectée
  void removeEdgeIfNotConnected(Edge edge);

  //! Définit la connectivité active pour le maillage associé
  /*! Ceci conditionne les connectivités à la charge de cette famille */
  void setConnectivity(const Integer c);

  void reorientEdgesIfNeeded();

 protected:

  ItemTypeInfo* m_edge_type = nullptr;
  bool m_has_edge = false;
  Integer m_node_prealloc = 0;
  Integer m_face_prealloc = 0;
  Integer m_cell_prealloc = 0;
  Integer m_mesh_connectivity = 0;
  NodeConnectivity* m_node_connectivity = nullptr;
  FaceConnectivity* m_face_connectivity = nullptr;
  CellConnectivity* m_cell_connectivity = nullptr;

 private:

  //! Famille des noeuds associée à cette famille
  NodeFamily* m_node_family = nullptr;

  inline void _removeEdge(Edge edge);
  inline void _createOne(ItemInternal* item,Int64 uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
