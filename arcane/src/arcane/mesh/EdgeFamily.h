// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeFamily.h                                                (C) 2000-2022 */
/*                                                                           */
/* Famille d'arêtes.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EDGEFAMILY_H
#define ARCANE_MESH_EDGEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemFamily.h"
#include "arcane/IItemFamilyModifier.h"
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
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, MeshInfos& mesh_info) override;
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily*  family() override {return this;}

  ItemInternal* allocOne(Int64 uid);
  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc);

  void replaceNode(ItemLocalId edge,Integer index,ItemLocalId node);

  //! \deprecated Utiliser la version removeCellFromEdge(ItemInternal*,ItemLocalId)
  void removeCellFromEdge(ItemInternal* edge,ItemInternal* cell_to_remove, bool no_destroy=false);

  //! Ajoute une maille voisine à une arête
  void addCellToEdge(Edge edge,Cell new_cell);
  //! Ajoute une maille voisine à une arête
  void addFaceToEdge(Edge edge,Face new_face);
  //! Supprime une maille d'une arête
  void removeCellFromEdge(Edge edge,ItemLocalId cell_to_remove_lid);
  //! Supprime une maille d'une arête
  ARCANE_DEPRECATED_260 void removeFaceFromEdge(ItemInternal* edge,ItemInternal* face_to_remove);
  //! Supprime une maille d'une arête
  void removeFaceFromEdge(ItemLocalId edge,ItemLocalId face_to_remove);
  //! Supprime l'arête si elle n'est plus connectée
  void removeEdgeIfNotConnected(Edge edge);

  //! Définit la connectivité active pour le maillage associé
  /*! Ceci conditionne les connectivités à la charge de cette famille */
  void setConnectivity(const Integer c);

 protected:

  ItemTypeInfo* m_edge_type;
  bool m_has_edge;
  Integer m_node_prealloc;
  Integer m_face_prealloc;
  Integer m_cell_prealloc;
  Integer m_mesh_connectivity;
  NodeConnectivity* m_node_connectivity;
  FaceConnectivity* m_face_connectivity;
  CellConnectivity* m_cell_connectivity;

 private:

  //! Famille des noeuds associée à cette famille
  NodeFamily* m_node_family;

  inline void _removeEdge(ItemInternal* edge);
  inline void _createOne(ItemInternal* item,Int64 uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
