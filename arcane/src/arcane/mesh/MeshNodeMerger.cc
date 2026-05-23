// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshNodeMerger.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Mesh node merger.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Connectivity.h"

#include "arcane/mesh/MeshNodeMerger.h"
#include "arcane/mesh/FaceReorienter.h"
#include "arcane/mesh/ItemTools.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/CellFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshNodeMerger::
MeshNodeMerger(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
  if (m_mesh->hasTiedInterface())
    throw NotImplementedException(A_FUNCINFO,"mesh with tied interfaces");
  if (m_mesh->dimension()==3){
    Int32 c = m_mesh->connectivity()();
    if (Connectivity::hasConnectivity(c,Connectivity::CT_HasEdge))
      throw NotImplementedException(A_FUNCINFO,"3D mesh with edges");
  }
  if (m_mesh->childMeshes().count()!=0)
    throw NotSupportedException(A_FUNCINFO,"mesh with child meshes");
  if (m_mesh->isAmrActivated())
    throw NotSupportedException(A_FUNCINFO,"mesh with AMR cells");

  m_node_family = ARCANE_CHECK_POINTER(dynamic_cast<NodeFamily*>(m_mesh->nodeFamily()));
  m_edge_family = ARCANE_CHECK_POINTER(dynamic_cast<EdgeFamily*>(m_mesh->edgeFamily()));
  m_face_family = ARCANE_CHECK_POINTER(dynamic_cast<FaceFamily*>(m_mesh->faceFamily()));
  m_cell_family = ARCANE_CHECK_POINTER(dynamic_cast<CellFamily*>(m_mesh->cellFamily()));

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * The principle of the algorithm is as follows:
 * 1. Determine the set of faces and cells that have at least one modified node.
 *    These are the ones that need to be processed.
 * 2. For faces, determine those whose nodes are all merged. These
 *    faces will be merged and disappear. It must be determined with which
 *    faces they will merge. In 3D, the same must be done for edges.
 * 3. Once everything is calculated, the connectivities of the
 *    entities (cells, faces, and edges) that are modified must be updated.
 * 4. Finally, the merged nodes, edges, and faces must be destroyed.
 *
 * It may be necessary to adapt this algorithm when the new
 * connectivities are in place.
 */
void MeshNodeMerger::
mergeNodes(Int32ConstArrayView nodes_local_id,
           Int32ConstArrayView nodes_to_merge_local_id,
           bool allow_non_corresponding_face)
{
  ItemInternalList nodes_internal(m_node_family->itemsInternal());
  Integer nb_node = nodes_local_id.size();
  if (nb_node != nodes_to_merge_local_id.size())
    throw ArgumentException(A_FUNCINFO,String::format("Arrays of different size"));
  for (Integer i = 0; i < nb_node; ++i) {
    Node node(nodes_internal[nodes_local_id[i]]);
    Node node_to_merge(nodes_internal[nodes_to_merge_local_id[i]]);
    // NOTE: June 2025: Remove the following test which is
    // not useful because it must be possible for a face to merge
    // a node with itself. The only case where this could cause a problem
    // with the current algorithm is if all these
    // nodes for a given face are merged with themselves.
    //  if (node.localId()==node_to_merge.localId())
    //    ARCANE_FATAL("Can not merge a node with itself");
    info(4) << "ADD CORRESPONDANCE node=" << node.uniqueId() << " node_to_merge=" << node_to_merge.uniqueId();
    m_nodes_correspondance.insert(std::make_pair(node_to_merge,node));
  }

  // Mark all faces that contain at least one merged node
  // and determine which ones must be merged: these are the ones for
  // which every node is merged.
  std::set<Face> marked_faces;
  Int64UniqueArray face_new_nodes_uid;
  Int64UniqueArray face_new_nodes_sorted_uid;
  ENUMERATE_ (Face, iface, m_face_family->allItems()) {
    Face face = *iface;
    Integer face_nb_node = face.nbNode();
    Integer nb_merged_node = 0;
    for( NodeEnumerator inode(face.nodes()); inode(); ++inode ){
      Node node = *inode;
      if (m_nodes_correspondance.find(node)!=m_nodes_correspondance.end()){
        ++nb_merged_node;
        marked_faces.insert(face);
      }
    }
    if (nb_merged_node == face_nb_node) {
      // All nodes of the face are merged. This means that the
      // cells associated with this face will reference a new face.
      // We must now find this new face.
      info(4) << "FACE TO MERGE uid=" << face.uniqueId();
      face_new_nodes_uid.resize(face_nb_node);
      face_new_nodes_sorted_uid.resize(face_nb_node);
      Node new_face_first_node;
      for( NodeEnumerator inode(face.nodes()); inode(); ++inode ){
        Node new_node = m_nodes_correspondance.find(*inode)->second;
        if (inode.index()==0)
          new_face_first_node = new_node;
        face_new_nodes_uid[inode.index()] = new_node.uniqueId();
        info(4) << " OLD_node=" << (*inode).uniqueId() << " new=" << new_node.uniqueId();
      }
      mesh_utils::reorderNodesOfFace(face_new_nodes_uid, face_new_nodes_sorted_uid);
      Face new_face = ItemTools::findFaceInNode2(new_face_first_node, face.type(), face_new_nodes_sorted_uid);
      if (new_face.null()) {
        // The face has no corresponding face. Do nothing if this is allowed.
        if (allow_non_corresponding_face)
          continue;
        ARCANE_FATAL("Can not find corresponding face nodes_uid={0}", face_new_nodes_sorted_uid);
      }
      info(4) << "NEW FACE=" << new_face.uniqueId() << " nb_cell=" << new_face.nbCell();
      m_faces_correspondance.insert(std::make_pair(face,new_face));
      // Since this face is merged, it is removed from the list of
      // marked faces.
      marked_faces.erase(marked_faces.find(face));
    }
  }
  // TODO: process edges

  // Mark all cells that contain at least one merged node.
  std::set<Cell> marked_cells;
  ENUMERATE_CELL(icell,m_cell_family->allItems()){
    Cell cell = *icell;
    for( NodeEnumerator inode(cell.nodes()); inode(); ++inode ){
      if (m_nodes_correspondance.find(*inode)!=m_nodes_correspondance.end())
        marked_cells.insert(cell);
    }
  }

  for( Cell cell : marked_cells ){
    ItemLocalId cell_local_id(cell.localId());
    info(4) << "MARKED CELL2=" << cell.localId();
    for( NodeEnumerator inode(cell.nodes()); inode(); ++inode ){
      Node node = *inode;
      auto x = m_nodes_correspondance.find(node);
      if (x!=m_nodes_correspondance.end()){
        Node new_node = x->second;
        info(4) << "REMOVE node=" << ItemPrinter(node) << " from cell=" << ItemPrinter(cell);
        m_node_family->removeCellFromNode(node,cell_local_id);
        m_node_family->addCellToNode(new_node,cell);
        m_cell_family->replaceNode(cell,inode.index(),new_node);
      }
    }
    for( FaceEnumerator iface(cell.faces()); iface(); ++iface ){
      Face face = *iface;
      auto x = m_faces_correspondance.find(face);
      if (x!=m_faces_correspondance.end()){
        Face new_face = x->second;
        m_face_family->removeCellFromFace(face,cell_local_id);
        if (new_face.backCell().null())
          m_face_family->addBackCellToFace(new_face,cell);
        else
          m_face_family->addFrontCellToFace(new_face,cell);
        m_cell_family->replaceFace(cell,iface.index(),new_face);
      }
    }
    // TODO: add edge management.
  }

  for( Face face : marked_faces ){
    info(4) << "MARKED FACE=" << face.localId();
    for( NodeEnumerator inode(face.nodes()); inode(); ++inode ){
      Node node = *inode;
      auto x = m_nodes_correspondance.find(node);
      if (x!=m_nodes_correspondance.end()){
        Node new_node = x->second;
        m_node_family->removeFaceFromNode(node,face);
        m_node_family->addFaceToNode(new_node,face);
        m_face_family->replaceNode(face,inode.index(),new_node);
      }
    }
  }
  // TODO: add edge management.

  // Ensure that the new faces are properly oriented
  {
    FaceReorienter fr(m_mesh);
    for( Face face : marked_faces ){
      fr.checkAndChangeOrientation(face);
    }
  }

  // Remove all faces that must be merged.
  for( const auto& x : m_faces_correspondance ){
    Face face = x.first;
    m_face_family->removeFaceIfNotConnected(face);
  }

  // Remove all nodes that must be merged.
  for( const auto& x : m_nodes_correspondance ){
    Node node = x.first;
    m_node_family->removeNodeIfNotConnected(node);
  }

  m_mesh->modifier()->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
