// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceFamily.h                                                (C) 2000-2024 */
/*                                                                           */
/* Famille de faces.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FACEFAMILY_H
#define ARCANE_MESH_FACEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/IItemFamilyModifier.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EdgeFamily;
class NodeFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille de faces.
 *
 * Cette classe gère une famille de face du maillage. La face a comme
 * particularité d'être orientée, et par conséquent elle possède une
 * maille dite derrière (Face::backCell()) et une maille devant
 * (Face::frontCell()).
 *
 * En général, une face n'est pas reliée à d'autres faces, sauf pour
 * les interfaces liées, où les faces esclaves ont une référence sur
 * la face maître correspondante.
 */
class ARCANE_MESH_EXPORT FaceFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class TopologyModifier;
  typedef ItemConnectivitySelectorT<NodeInternalConnectivityIndex,IncrementalItemConnectivity> NodeConnectivity;
  typedef ItemConnectivitySelectorT<EdgeInternalConnectivityIndex,IncrementalItemConnectivity> EdgeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex,IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex,IncrementalItemConnectivity> CellConnectivity;
  typedef ItemConnectivitySelectorT<HParentInternalConnectivityIndex, IncrementalItemConnectivity> HParentConnectivity;
  typedef ItemConnectivitySelectorT<HChildInternalConnectivityIndex, IncrementalItemConnectivity> HChildConnectivity;

 public:

  FaceFamily(IMesh* mesh,const String& name);
  virtual ~FaceFamily(); //<! Libère les ressources

 public:

  void build() override;
  virtual void preAllocate(Integer nb_item);

 public:

  // IItemFamilyModifier Interface
  Item allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info) override;
  Item findOrAllocOne(Int64 uid,ItemTypeId type_id,MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily* family() override { return this; }

  // TODO: DEPRECATED
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type);
  // TODO: DEPRECATED
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,bool& is_alloc);

  Face allocOne(Int64 uid,ItemTypeId type);
  Face findOrAllocOne(Int64 uid,ItemTypeId type,bool& is_alloc);

 public:

  //! Ajoute une maille derrière la face
  void addBackCellToFace(Face face,Cell new_cell);
  //! Ajoute une maille devant la face
  void addFrontCellToFace(Face face,Cell new_cell);
  //! Supprime une maille de la face
  void removeCellFromFace(Face face,ItemLocalId cell_to_remove_lid);
  //! Ajoute une arête devant la face
  void addEdgeToFace(Face face,Edge new_edge);
  //! Supprime une arête de la face
  /*! Pas de notion de no_destroy car la consistence est orientée par les mailles et non les arêtes */
  void removeEdgeFromFace(Face face,Edge edge_to_remove);
  //! Supprime la face si elle n'est plus connectée
  void removeFaceIfNotConnected(Face face);

  void replaceNode(ItemLocalId face,Integer index,ItemLocalId node);
  void replaceEdge(ItemLocalId face,Integer index,ItemLocalId edge);
  void replaceFace(ItemLocalId face,Integer index,ItemLocalId face2);
  void replaceCell(ItemLocalId face,Integer index,ItemLocalId cell);

  void setBackAndFrontCells(Face face,Int32 back_cell_lid,Int32 front_cell_lid);

  //! AMR
  void replaceBackCellToFace(Face face,ItemLocalId new_cell);
  void replaceFrontCellToFace(Face face,ItemLocalId new_cell);
  void addBackFrontCellsFromParentFace(Face subface,Face face);
  void replaceBackFrontCellsFromParentFace(Cell subcell,Face subface,Cell cell,Face face);
  bool isSubFaceInFace(Face subface,Face face) const;
  bool isChildOnFace(ItemWithNodes child,Face face) const;
  void subFaces(Face face,Array<ItemInternal*>& subfaces);
  void allSubFaces(Face face,Array<ItemInternal*>& subfaces);
  void activeSubFaces(Face face,Array<ItemInternal*>& subfaces);
  void familyTree (Array<ItemInternal*>& family,Cell item, const bool reset=true) const;
  void activeFamilyTree (Array<ItemInternal*>& family,Cell item, const bool reset=true) const;
  void _addChildFaceToFace(Face parent_face, Face child_face);
  void _addParentFaceToFace(Face parent_face, Face child_face);
  // OFF AMR

  /*!
   * \brief Indique s'il faut vérifier l'orientation des mailles et des faces.
   *
   * Normalement, cette option doit être active. Cependant, il est possible
   * dans certains cas, comme lors d'un raffinement, que l'orientation
   * ne soit pas correcte. Par exemple, il est possible d'avoir deux mailles
   * derrière une face. Dans ce cas, il faut désactiver cette option.
   */
  void setCheckOrientation(bool is_check) { m_check_orientation = is_check; }

  //! Renseigne les informations liées à l'interface liée \a interface
  void applyTiedInterface(ITiedInterface* interface);

  //! Supprime les informations liées à l'interface liée \a interface
  void removeTiedInterface(ITiedInterface* interface);

  void setConnectivity(const Integer c);

  void reorientFacesIfNeeded();

 public:

  virtual void computeSynchronizeInfos() override;

 private:

  Integer m_node_prealloc = 0;
  Integer m_edge_prealloc = 0;
  Integer m_cell_prealloc = 0;
  Integer m_mesh_connectivity = 0;

  //! Famille des noeuds associée à cette famille
  NodeFamily* m_node_family = nullptr;

  //! Famille d'arêtes associée à cette famille
  EdgeFamily* m_edge_family = nullptr;

  //! Indique s'il faut vérifier l'orientation
  bool m_check_orientation = true;

  NodeConnectivity* m_node_connectivity = nullptr;
  EdgeConnectivity* m_edge_connectivity = nullptr;
  FaceConnectivity* m_face_connectivity = nullptr;
  CellConnectivity* m_cell_connectivity = nullptr;
  HParentConnectivity* m_hparent_connectivity = nullptr;
  HChildConnectivity* m_hchild_connectivity = nullptr;

  bool m_has_face = true;

 private:

  void _addMasterFaceToFace(Face face,Face master_face);
  void _addSlaveFacesToFace(Face face,Int32ConstArrayView slave_faces_lid);
  void _removeMasterFaceToFace(Face face);
  void _removeSlaveFacesToFace(Face face);

  inline void _removeFace(Face face);
  Real3 _computeFaceNormal(Face face, const SharedVariableNodeReal3& nodes_coord) const;
  inline void _createOne(ItemInternal* item,Int64 uid,ItemTypeInfo* type);
  inline void _createOne(ItemInternal* item,Int64 uid,ItemTypeId type_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
