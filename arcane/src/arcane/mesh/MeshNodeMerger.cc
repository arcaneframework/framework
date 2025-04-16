// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshNodeMerger.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Fusions de noeuds d'un maillage.                                          */
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
 * Le principe de l'algorithme est le suivant:
 * 1. Détermine l'ensemble des faces et des mailles qui ont au moins un noeud modifié.
 *    Ce sont celles qu'il faudra traiter.
 * 2. Pour les faces, détermine celles qui ont tous leur noeuds fusionnées. Ces
 *    faces seront fusionnées et disparaitrons. Il faut déterminer avec quelle
 *    faces elles vont fusionner. En 3D, il faudra faire de même pour les arêtes.
 * 3. Une fois tout calculé, il faut mettre à jour les connectivités des
 *    entités (mailles, faces et arêtes) qui sont modifiées.
 * 4. Enfin, il faut détruire les noeuds, arêtes et faces fusionnées.
 *
 * Il faudra éventuellement adapter cet algorithme lorsque les nouvelles
 * connectivités seront en place.
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
    if (node.localId()==node_to_merge.localId())
      ARCANE_FATAL("Can not merge a node with itself");
    info(4) << "ADD CORRESPONDANCE node=" << node.uniqueId() << " node_to_merge=" << node_to_merge.uniqueId();
    m_nodes_correspondance.insert(std::make_pair(node_to_merge,node));
  }

  // Marque toutes les faces qui contiennent au moins un nœud fusionné
  // et détermine celles qui doivent être fusionnées : ce sont celles pour
  // lesquelles chaque nœud est fusionné.
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
      // Tous les nœuds de la face sont fusionnés. Cela veut dire que les
      // mailles associées à cette face vont faire référence à une nouvelle face.
      // Il faut maintenant trouver cette nouvelle face.
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
        // La face n'a pas de correspondante. Ne fais rien si cela est autorisé.
        if (allow_non_corresponding_face)
          continue;
        ARCANE_FATAL("Can not find corresponding face nodes_uid={0}", face_new_nodes_sorted_uid);
      }
      info(4) << "NEW FACE=" << new_face.uniqueId() << " nb_cell=" << new_face.nbCell();
      m_faces_correspondance.insert(std::make_pair(face,new_face));
      // Comme cette face est fusionnée, on la retire de la liste des faces
      // marquées.
      marked_faces.erase(marked_faces.find(face));
    }
  }
  // TODO: traiter les arêtes

  // Marque toutes les mailles qui contiennent au moins un noeud fusionné.
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
    // TODO: ajouter gestion des aretes.
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
  // TODO: ajouter gestion des arêtes.

  // S'assure que les nouvelles faces sont bien orientées
  {
    FaceReorienter fr(m_mesh);
    for( Face face : marked_faces ){
      fr.checkAndChangeOrientation(face);
    }
  }

  // Supprime toutes les faces qui doivent être fusionnées.
  for( const auto& x : m_faces_correspondance ){
    Face face = x.first;
    m_face_family->removeFaceIfNotConnected(face);
  }

  // Supprime tous les noeuds qui doivent être fusionnées.
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

