// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeFamily.h                                                (C) 2000-2025 */
/*                                                                           */
/* Node family.                                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_NODEFAMILY_H
#define ARCANE_MESH_NODEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemFamilyModifier.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/MeshInfos.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Node family.
 */
class ARCANE_MESH_EXPORT NodeFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class ItemCompare2;
  class ItemCompare3;
  class TopologyModifier;

  typedef ItemConnectivitySelectorT<EdgeInternalConnectivityIndex,IncrementalItemConnectivity> EdgeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex,IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex,IncrementalItemConnectivity> CellConnectivity;
  typedef ItemConnectivitySelectorT<HParentInternalConnectivityIndex, IncrementalItemConnectivity> HParentConnectivity;
  typedef ItemConnectivitySelectorT<HChildInternalConnectivityIndex, IncrementalItemConnectivity> HChildConnectivity;

 public:

  NodeFamily(IMesh* mesh,const String& name);
  virtual ~NodeFamily(); //<! Releases resources

 public:

  void build() override;
  void computeSynchronizeInfos() override;
  void _endAllocate() override;

 public:

  void preAllocate(Integer nb_item);

 public:

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*!
   * \brief Allocates a face with a unique number \a uid and type \a type.
   *
   * This version is designed to be called in a generic block that ignores the type
   * of the item. The update of the mesh's item count is therefore done in this method,
   * and not in the calling block.
   */

  // IItemFamilyModifier interface
  Item allocOne(Int64 uid,ItemTypeId type_id, MeshInfos& mesh_info) override
  {
    ARCANE_ASSERT((type_id == IT_Vertex),("Creating node with a type not equal to IT_Vertex"));
    ARCANE_UNUSED(type_id);
    ++mesh_info.nbNode();
    return allocOne(uid);
  }

  ItemInternal* allocOne(Int64 uid)
  {
    ItemInternal* item = _allocOne(uid);
    m_item_internal_list->nodes = _itemsInternal();
    _allocateInfos(item,uid,m_node_type);
    return item;
  }

  // IItemFamilyModifier interface
  Item findOrAllocOne(Int64 uid,ItemTypeId type_id,MeshInfos& mesh_info, bool& is_alloc) override
  {
    ARCANE_ASSERT((type_id == IT_Vertex),("Creating node with a type not equal to IT_Vertex"));
    ARCANE_UNUSED(type_id);
    auto node = findOrAllocOne(uid,is_alloc);
    if (is_alloc)
      ++mesh_info.nbNode();
    return node;
  }

  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc)
  {
    ItemInternal* item = _findOrAllocOne(uid,is_alloc);
    if (is_alloc){
      m_item_internal_list->nodes = _itemsInternal();
      _allocateInfos(item,uid,m_node_type);
    }
    return item;
  }

  // IItemFamilyModifier interface
  IItemFamily* family() override {return this;}

  void replaceEdge(ItemLocalId node,Integer index,ItemLocalId edge);
  void replaceFace(ItemLocalId node,Integer index,ItemLocalId face);
  void replaceCell(ItemLocalId node,Integer index,ItemLocalId cell);

  void addCellToNode(Node node,Cell new_cell);
  void addFaceToNode(Node node,Face new_face);
  void addEdgeToNode(Node node,Edge new_edge);
  void removeEdgeFromNode(ItemLocalId node,ItemLocalId edge_to_remove);
  void removeFaceFromNode(ItemLocalId node,ItemLocalId face_to_remove);
  void removeCellFromNode(Node node,ItemLocalId cell_to_remove_lid);
  //! Removes the node if it is no longer connected
  void removeNodeIfNotConnected(Node node);

  VariableNodeReal3& nodesCoordinates()
  {
    ARCANE_ASSERT((m_nodes_coords),("NodeFamily::nodesCoordinates is available only on primary meshes"));
    return *m_nodes_coords;
  }

  //! Sets the active connectivity for the associated mesh
  /*! This conditions the connectivities to the charge of this family */
  void setConnectivity(const Integer c);

  void sortInternalReferences();

  void notifyItemsUniqueIdChanged() override;

  void _addChildNodeToNode(Node parent_node, Node child_node);
  void _addParentNodeToNode(Node parent_node, Node child_node);

 private:
  
  ItemTypeInfo* m_node_type = nullptr; //!< Instance containing the node type
  Integer m_edge_prealloc = 0;
  Integer m_face_prealloc = 0;
  Integer m_cell_prealloc = 0;
  Integer m_mesh_connectivity = 0;
  bool m_no_face_connectivity = false;
  VariableNodeReal3* m_nodes_coords = nullptr;
  EdgeConnectivity* m_edge_connectivity = nullptr;
  FaceConnectivity* m_face_connectivity = nullptr;
  CellConnectivity* m_cell_connectivity = nullptr;
  HParentConnectivity* m_hparent_connectivity = nullptr;
  HChildConnectivity* m_hchild_connectivity = nullptr;

  FaceFamily* m_face_family = nullptr;
  EdgeFamily* m_edge_family = nullptr;
  //! Indicates if connected faces and edges are sorted by nodes
  bool m_is_sort_connected_faces_and_edges = false;

 private:

  inline void _removeNode(Node node);
  void _sortConnectedItems(IItemFamily* family, IncrementalItemConnectivity* connectivity);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
