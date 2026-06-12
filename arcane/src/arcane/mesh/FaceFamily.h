// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceFamily.h                                                (C) 2000-2024 */
/*                                                                           */
/* Face family.                                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FACEFAMILY_H
#define ARCANE_MESH_FACEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IItemFamilyModifier.h"

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
 * \brief Face family.
 *
 * This class manages a face family of the mesh. The face has the
 * characteristic of being oriented, and consequently, it possesses a
 * so-called back cell (Face::backCell()) and a front cell
 * (Face::frontCell()).
 *
 * Generally, a face is not connected to other faces, except for
 * tied interfaces, where the slave faces have a reference to
 * the corresponding master face.
 */
class ARCANE_MESH_EXPORT FaceFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class TopologyModifier;
  typedef ItemConnectivitySelectorT<NodeInternalConnectivityIndex, IncrementalItemConnectivity> NodeConnectivity;
  typedef ItemConnectivitySelectorT<EdgeInternalConnectivityIndex, IncrementalItemConnectivity> EdgeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex, IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex, IncrementalItemConnectivity> CellConnectivity;
  typedef ItemConnectivitySelectorT<HParentInternalConnectivityIndex, IncrementalItemConnectivity> HParentConnectivity;
  typedef ItemConnectivitySelectorT<HChildInternalConnectivityIndex, IncrementalItemConnectivity> HChildConnectivity;

 public:

  FaceFamily(IMesh* mesh, const String& name);
  virtual ~FaceFamily(); //<! Frees resources

 public:

  void build() override;
  virtual void preAllocate(Integer nb_item);

 public:

  // IItemFamilyModifier Interface
  Item allocOne(Int64 uid, ItemTypeId type_id, MeshInfos& mesh_info) override;
  Item findOrAllocOne(Int64 uid, ItemTypeId type_id, MeshInfos& mesh_info, bool& is_alloc) override;
  IItemFamily* family() override { return this; }

  // TODO: DEPRECATED
  ItemInternal* allocOne(Int64 uid, ItemTypeInfo* type);
  // TODO: DEPRECATED
  ItemInternal* findOrAllocOne(Int64 uid, ItemTypeInfo* type, bool& is_alloc);

  Face allocOne(Int64 uid, ItemTypeId type);
  Face findOrAllocOne(Int64 uid, ItemTypeId type, bool& is_alloc);

 public:

  //! Adds a back cell to the face
  void addBackCellToFace(Face face, Cell new_cell);
  //! Adds a front cell to the face
  void addFrontCellToFace(Face face, Cell new_cell);
  //! Removes a cell from the face
  void removeCellFromFace(Face face, ItemLocalId cell_to_remove_lid);
  //! Adds an edge to the face
  void addEdgeToFace(Face face, Edge new_edge);
  //! Removes an edge from the face
  /*! No notion of no_destroy because the consistency is determined by the cells and not the edges */
  void removeEdgeFromFace(Face face, Edge edge_to_remove);
  //! Removes the face if it is no longer connected
  void removeFaceIfNotConnected(Face face);

  void replaceNode(ItemLocalId face, Integer index, ItemLocalId node);
  void replaceEdge(ItemLocalId face, Integer index, ItemLocalId edge);
  void replaceFace(ItemLocalId face, Integer index, ItemLocalId face2);
  void replaceCell(ItemLocalId face, Integer index, ItemLocalId cell);

  void setBackAndFrontCells(Face face, Int32 back_cell_lid, Int32 front_cell_lid);

  //! AMR
  void replaceBackCellToFace(Face face, ItemLocalId new_cell);
  void replaceFrontCellToFace(Face face, ItemLocalId new_cell);
  void addBackFrontCellsFromParentFace(Face subface, Face face);
  void replaceBackFrontCellsFromParentFace(Cell subcell, Face subface, Cell cell, Face face);
  bool isSubFaceInFace(Face subface, Face face) const;
  bool isChildOnFace(ItemWithNodes child, Face face) const;
  void subFaces(Face face, Array<ItemInternal*>& subfaces);
  void allSubFaces(Face face, Array<ItemInternal*>& subfaces);
  void activeSubFaces(Face face, Array<ItemInternal*>& subfaces);
  void familyTree(Array<ItemInternal*>& family, Cell item, const bool reset = true) const;
  void activeFamilyTree(Array<ItemInternal*>& family, Cell item, const bool reset = true) const;
  void _addChildFaceToFace(Face parent_face, Face child_face);
  void _addParentFaceToFace(Face parent_face, Face child_face);
  // OFF AMR

  /*!
   * \brief Indicates whether the orientation of the cells and faces must be checked.
   *
   * Normally, this option must be active. However, it is possible
   * in certain cases, such as during refinement, that the orientation
   * is not correct. For example, it is possible to have two cells
   * behind a face. In this case, this option must be deactivated.
   */
  void setCheckOrientation(bool is_check) { m_check_orientation = is_check; }

  //! Sets the information related to the tied interface \a interface
  void applyTiedInterface(ITiedInterface* interface);

  //! Removes the information related to the tied interface \a interface
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

  //! Node family associated with this family
  NodeFamily* m_node_family = nullptr;

  //! Edge family associated with this family
  EdgeFamily* m_edge_family = nullptr;

  //! Indicates whether to check the orientation
  bool m_check_orientation = true;

  NodeConnectivity* m_node_connectivity = nullptr;
  EdgeConnectivity* m_edge_connectivity = nullptr;
  FaceConnectivity* m_face_connectivity = nullptr;
  CellConnectivity* m_cell_connectivity = nullptr;
  HParentConnectivity* m_hparent_connectivity = nullptr;
  HChildConnectivity* m_hchild_connectivity = nullptr;

  bool m_has_face = true;

 private:

  void _addMasterFaceToFace(Face face, Face master_face);
  void _addSlaveFacesToFace(Face face, Int32ConstArrayView slave_faces_lid);
  void _removeMasterFaceToFace(Face face);
  void _removeSlaveFacesToFace(Face face);

  inline void _removeFace(Face face);
  Real3 _computeFaceNormal(Face face, const SharedVariableNodeReal3& nodes_coord) const;
  inline void _createOne(ItemInternal* item, Int64 uid, ItemTypeInfo* type);
  inline void _createOne(ItemInternal* item, Int64 uid, ItemTypeId type_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
