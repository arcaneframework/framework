// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshIncrementalBuilder.cc                            (C) 2000-2025 */
/*                                                                           */
/* Construction d'un maillage de manière incrémentale.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iterator.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IItemFamilyModifier.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"
#include "arcane/mesh/GhostLayerBuilder.h"
#include "arcane/mesh/FaceUniqueIdBuilder.h"
#include "arcane/mesh/EdgeUniqueIdBuilder.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/GraphDoFs.h"
#include "arcane/mesh/ParticleFamily.h"

#include <set>
#include <map>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_DYNAMIC_MESH
// #define ARCANE_DEBUG_DYNAMIC_MESH2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshIncrementalBuilder::
DynamicMeshIncrementalBuilder(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_item_type_mng(mesh->itemTypeMng())
, m_has_amr(mesh->isAmrActivated())
, m_one_mesh_item_adder(new OneMeshItemAdder(this))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_VERBOSE_MESH_BUILDER", true))
    m_verbose = (v.value()!=0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshIncrementalBuilder::
~DynamicMeshIncrementalBuilder()
{
  delete m_one_mesh_item_adder;
  delete m_face_unique_id_builder;
  delete m_edge_unique_id_builder;
  delete m_ghost_layer_builder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_printCellFaceInfos(ItemInternal* icell,const String& str)
{
  Cell cell(icell);
  Integer nb_num_back_face = 0;
  Integer nb_num_front_face = 0;
  Integer index = 0;
  for( Face face : cell.faces() ){
    String add_msg;
    if (face.backCell()==cell){
      ++nb_num_back_face;
      add_msg = "is a back face";
    }
    else{
      ++nb_num_front_face;
      add_msg = "is a front face";
    }
    info() << str << ": celluid=" << cell.uniqueId()
           << " faceuid=" << face.uniqueId()
           << " faceindex=" << index
           << add_msg;
    ++index;
  }
  info() << str << ": celluid=" << cell.uniqueId()
         << " nbback=" << nb_num_back_face
         << " nbfront=" << nb_num_front_face;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des mailles au maillage actuel.
 *
 * \param mesh_nb_cell nombre de mailles à ajouter
 * \param cells_infos infos sur les maillage (voir IMesh::allocateMesh())
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 * \param cells en retour, si non vide, contient les mailles créées.
 */
void DynamicMeshIncrementalBuilder::
addCells(Integer nb_cell,Int64ConstArrayView cells_infos,
         Integer sub_domain_id,Int32ArrayView cells,
         bool allow_build_face)
{
  ItemTypeMng* itm = m_item_type_mng;

  debug() << "[addCells] ADD CELLS mesh=" << m_mesh->name() << " nb=" << nb_cell;
  Integer cells_infos_index = 0;
  bool add_to_cells = cells.size()!=0;
  if (add_to_cells && nb_cell!=cells.size())
    ARCANE_THROW(ArgumentException,"return array 'cells' has to have same size as number of cells");
  for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
    ItemTypeId item_type_id { (Int16)cells_infos[cells_infos_index] };
    ++cells_infos_index;
    Int64 cell_unique_id = cells_infos[cells_infos_index];
    ++cells_infos_index;

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_cell_nb_node = it->nbLocalNode();
    Int64ConstArrayView current_cell_nodes_uid(current_cell_nb_node,&cells_infos[cells_infos_index]);
    
    ItemInternal* cell = m_one_mesh_item_adder->addOneCell(item_type_id,cell_unique_id,sub_domain_id,current_cell_nodes_uid,
                                                           allow_build_face);
    
    if (add_to_cells)
      cells[i_cell] = cell->localId();
    cells_infos_index += current_cell_nb_node;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des mailles au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param mesh_nb_cell nombre de mailles à ajouter
 * \param cells_infos infos sur les maillage (voir IMesh::allocateMesh())
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 * \param cells en retour, si non vide, contient les mailles créées.
 */

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addCells3(Integer nb_cell,Int64ConstArrayView cells_infos,
          Integer sub_domain_id,Int32ArrayView cells,
          bool allow_build_face)
{
  bool add_to_cells = cells.size()!=0;
  if (add_to_cells && nb_cell!=cells.size())
    throw ArgumentException(A_FUNCINFO,
        "return array 'cells' has to have same size as number of cells");
  ItemDataList item_data_list; // store item ids and dependency connectivities
  ItemDataList item_relation_data_list; // store relation connectivities
  ItemData& cell_data = item_data_list.itemData(Integer(m_mesh->cellFamily()->itemKind()),
      nb_cell,0,cells,&mesh()->trueCellFamily(),&mesh()->trueCellFamily(),sub_domain_id);
  // item info_size cannot be guessed
  Int64UniqueArray faces_infos, edges_infos, nodes_infos, node_uids;
  Int32UniqueArray face_lids, edge_lids, node_lids;
  Integer nb_face = 0 , nb_edge = 0, nb_node = 0;
  std::map<Int64, Int64SharedArray> cell_to_face_connectivity_info;
  std::map<std::pair<Int64,Int64>, Int64> edge_uid_map;

  // Fill classical info
  _fillFaceInfo(nb_face, nb_cell, faces_infos, cells_infos, cell_to_face_connectivity_info);
  _fillEdgeInfo(nb_edge, nb_face, edges_infos, faces_infos, edge_uid_map);
  _fillNodeInfo(nb_node, nb_face, nodes_infos, faces_infos);

  // Fill Cell info 2
  _fillCellNewInfoNew(nb_cell,cells_infos,cell_data.itemInfos(),cell_to_face_connectivity_info, edge_uid_map);
  // Fill Face info 2
  if (allow_build_face) {
    ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),nb_face,0,face_lids,&mesh()->trueFaceFamily(),&mesh()->trueFaceFamily(),sub_domain_id);
    face_lids.resize(item_data_list[m_mesh->faceFamily()->itemKind()].nbItems());
    _fillFaceNewInfoNew(nb_face,faces_infos,face_data.itemInfos(),edge_uid_map);// todo voir appel depuis add Faces
    ItemData& face_relation_data = item_relation_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),nb_face,0,face_lids,&mesh()->trueFaceFamily(),&mesh()->trueFaceFamily(),sub_domain_id);
    _initFaceRelationInfo(face_relation_data,cell_data, faces_infos); // face-cell relations
  }
  // Fill Edge info 2
  if (hasEdge()) {
    Integer edge_info_size = 1 + nb_edge*6;//New info = Nb_connected_family + for each edge : type, uid, connected_family_kind, nb_node_connected, first_node_uid, second_node_uid
    ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),nb_edge,edge_info_size,edge_lids,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);
    edge_lids.resize(item_data_list[m_mesh->edgeFamily()->itemKind()].nbItems());
    _fillEdgeNewInfoNew(nb_edge,edges_infos,edge_data.itemInfos()); // reprendre fillEdgeInfo et voir l'appel depuis add Edges + attention ordre des noeuds,
    ItemData& edge_relation_data = item_relation_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),nb_edge,edge_info_size,edge_lids,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);
    _initEdgeRelationInfo(edge_relation_data, cell_data, edges_infos); // edge-cell relations
    if (allow_build_face) {
        _appendEdgeRelationInfo(edge_relation_data,item_data_list[IK_Face], edges_infos); // edge-face relations
        _appendFaceRelationInfo(item_relation_data_list[IK_Face],edge_data,faces_infos); // face-edge relations
    }
  }
  // Fill Node info 2
  Integer node_info_size = 1 + nb_node*2;//New info = Nb_connected_family + for each node : type, uid (no connected family)
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,node_info_size,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  node_lids.resize(item_data_list[m_mesh->nodeFamily()->itemKind()].nbItems());
  _fillNodeNewInfo(nodes_infos,node_data.itemInfos());
  ItemData& node_relation_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,node_info_size,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  _initNodeRelationInfo(node_relation_data, cell_data, nodes_infos); // node-cell relations
  if (allow_build_face)_appendNodeRelationInfo(node_relation_data, item_data_list[IK_Face], nodes_infos); // node-face relations
  if (hasEdge())_appendNodeRelationInfo(node_relation_data, item_data_list[IK_Edge], nodes_infos); // node-edge relations

  // Launch node, face and edges (not mandatory) and cell creation
  addItems(item_data_list, item_relation_data_list);

}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addCells2(Integer nb_cell,Int64ConstArrayView cells_infos,
          Integer sub_domain_id,Int32ArrayView cells,
          bool allow_build_face)
{
  ItemDataList item_data_list;
  ItemData& cell_data = item_data_list.itemData(Integer(m_mesh->cellFamily()->itemKind()),
      nb_cell,0,cells,&mesh()->trueCellFamily(),&mesh()->trueCellFamily(),sub_domain_id);
  // item info_size cannot be guessed
  Int64Array& cell_infos2 = cell_data.itemInfos();
  Int64UniqueArray faces_infos;
  Int64UniqueArray node_uids;
  Int32UniqueArray face_lids;
  Integer nb_face;


  // Fill cell info 2
  _fillCellInfo2(nb_cell,cells_infos,cell_infos2,nb_face, faces_infos,node_uids,allow_build_face);

  // Fill face info 2
  if (allow_build_face)
    {
      Integer face_infos2_size = faces_infos.size()+ 1 + 2 * nb_face; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
      face_lids.resize(nb_face);
      ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),
          nb_face,face_infos2_size,face_lids,&mesh()->trueFaceFamily(),&mesh()->trueFaceFamily(),sub_domain_id);
      Int64ArrayView face_infos2 = face_data.itemInfos().view();
      _fillFaceInfo2(nb_face,faces_infos,face_infos2,node_uids); // if allow_build_face false, node_uids built in fillCellInfo3
    }

  // Fill node info 2
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
      nb_node_to_add,2*nb_node_to_add+1,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids,node_infos2);

  // Launch node and face and cell creation
  addItems(item_data_list);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceInfo(Integer& nb_face, Integer nb_cell,Int64Array& faces_infos, Int64ConstArrayView cells_infos, std::map<Int64,Int64SharedArray>& cell_to_face_connectivity_info)
{
  faces_infos.reserve(2*cells_infos.size());
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  nb_face = 0;
  NodeInFaceSet face_nodes_set;
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    Integer item_type_id = (Integer)cells_infos[cells_infos_index++];
    Int64 current_cell_unique_id = cells_infos[cells_infos_index++];
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_cell_nb_node = it->nbLocalNode();
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node,&cells_infos[cells_infos_index]);
    Integer current_cell_nb_face = it->nbLocalFace();
    for (Integer face_in_cell_index = 0; face_in_cell_index < current_cell_nb_face; ++face_in_cell_index)
      {
        const ItemTypeInfo::LocalFace& current_face = it->localFace(face_in_cell_index);
        Integer current_face_nb_node = current_face.nbNode();
        work_face_orig_nodes.resize(current_face_nb_node);
        work_face_sorted_nodes.resize(current_face_nb_node);
        Integer i = 0;
        for (Integer node_in_face_index = 0; node_in_face_index < current_face_nb_node; ++node_in_face_index)
          {
            work_face_orig_nodes[i++] = current_cell_node_uids[current_face.node(node_in_face_index)];
          }
        mesh_utils::reorderNodesOfFace(work_face_orig_nodes,work_face_sorted_nodes);
        // get face uid: we shouldn't have to guess, it should be given in arguments...
        Int64 current_face_uid = _findFaceUniqueId(work_face_sorted_nodes,face_nodes_set);
        if (current_face_uid == NULL_ITEM_ID) // face not already known add to face info
          {
            current_face_uid = m_face_uid_pool++;
            faces_infos.add(current_face.typeId()); // face info type
            faces_infos.add(current_face_uid); // face info uid
            faces_infos.addRange(work_face_sorted_nodes); // face info node uids
            _addFaceNodes(face_nodes_set,work_face_sorted_nodes,current_face_uid);
            ++nb_face;
          }
        cell_to_face_connectivity_info[current_cell_unique_id].add(current_face_uid);
      }
    cells_infos_index += current_cell_nb_node;
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeInfo(Integer& nb_edge, Integer nb_face,Int64Array& edges_infos,
              Int64ConstArrayView faces_infos,std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map)
{
  edges_infos.reserve(2*faces_infos.size());
  ItemTypeMng* itm = m_item_type_mng;
  Integer faces_infos_index = 0;
  nb_edge = 0;
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {
    Integer item_type_id = (Integer)faces_infos[faces_infos_index++]; // face type
    ++faces_infos_index;// face uid (unused)
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_face_nb_node = it->nbLocalNode();
    Int64ConstArrayView current_face_node_uids(current_face_nb_node,&faces_infos[faces_infos_index]);
    Integer current_face_nb_edge = it->nbLocalEdge();
    for (Integer edge_in_face_index = 0; edge_in_face_index < current_face_nb_edge; ++edge_in_face_index)
      {
        const ItemTypeInfo::LocalEdge& current_edge = it->localEdge(edge_in_face_index);
        Int64 first_node  = current_face_node_uids[current_edge.beginNode()];
        Int64 second_node = current_face_node_uids[current_edge.endNode()];
        if (first_node > second_node) std::swap(first_node,second_node);
        auto edge_it = edge_uid_map.insert(std::make_pair(std::make_pair(first_node,second_node), m_edge_uid_pool));
        if (edge_it.second)
          {
            edges_infos.add(m_edge_uid_pool); // edge uid
            edges_infos.add(first_node); // first node uid
            edges_infos.add(second_node); // second node uid
            ++nb_edge;
            ++m_edge_uid_pool;
          }
      }
    faces_infos_index += current_face_nb_node;
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeInfo(Integer& nb_node, Integer nb_face,Int64Array& nodes_infos, Int64ConstArrayView faces_infos)
{
  nodes_infos.reserve(faces_infos.size());
  Integer faces_infos_index = 0;
  std::set<Int64> nodes_set;
  ItemTypeMng* itm = m_item_type_mng;
  for (Integer i_face = 0; i_face < nb_face; ++i_face) {
    Int32 type_id = CheckedConvert::toInt32(faces_infos[faces_infos_index++]);
    Integer current_face_nb_node = itm->typeFromId(type_id)->nbLocalNode(); // face type
    ++faces_infos_index;//face_uid (unused)
    for (auto node_uid : faces_infos.subConstView(faces_infos_index,current_face_nb_node)) {
        nodes_set.insert(node_uid);
    }
    faces_infos_index+=current_face_nb_node;
  }
  nb_node = CheckedConvert::toInteger(nodes_set.size());
  for (auto node_uid : nodes_set) {
    nodes_infos.add(node_uid);
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeInfoFromEdge(Integer& nb_node, Integer nb_edge, Int64Array& nodes_infos, Int64ConstArrayView edges_infos)
{
  nodes_infos.reserve(2*edges_infos.size());
  Integer edges_infos_index = 0;
  std::set<Int64> nodes_set;
  for (Integer i_edge = 0; i_edge < nb_edge; ++i_edge) {
    ++edges_infos_index;//edge_uid (unused)
    nodes_set.insert(edges_infos[edges_infos_index++]);
    nodes_set.insert(edges_infos[edges_infos_index++]);
  }
  nb_node = CheckedConvert::toInteger(nodes_set.size());
  for (auto node_uid : nodes_set) {
    nodes_infos.add(node_uid);
  }
}
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillCellNewInfoNew(Integer nb_cell,Int64ConstArrayView cells_infos,Int64Array& cell_infos2,
                    const std::map<Int64,Int64SharedArray>& cell_to_face_connectivity_info, const std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map)
{
  Integer cell_infos2_size_approx = 1 + 3 * (cells_infos.size()+ 2*nb_cell); // supposes as many faces as nodes as edges...
  cell_infos2.reserve(cell_infos2_size_approx);
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  Integer nb_connected_families = hasEdge() ? 3 : 2;
  cell_infos2.add(nb_connected_families); // nb_connected_families (node and face +/- edge)
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {
    Integer item_type_id = (Integer)cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell type
    Int64 current_cell_uid = cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell uid
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    //--- Cell node connectivity
    cell_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id: node family
    Integer current_cell_nb_node = it->nbLocalNode();
    cell_infos2.add(current_cell_nb_node); // nb_connected_nodes
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node,&cells_infos[cells_infos_index]);
    cell_infos2.addRange(current_cell_node_uids); // node ids
    //--- Cell face connectivity
    cell_infos2.add(mesh()->faceFamily()->itemKind()); // connected_family_id: face family
    Integer current_cell_nb_face = it->nbLocalFace();
    cell_infos2.add(current_cell_nb_face); // nb_connected_faces
    Int64ArrayView current_cell_faces = cell_to_face_connectivity_info.find(current_cell_uid)->second.view();
    if (current_cell_nb_face != current_cell_faces.size())
      ARCANE_FATAL("Incoherent number of faces for cell {0}. Expected {1} found {2}",
                   current_cell_uid,current_cell_nb_face,current_cell_faces.size());
    cell_infos2.addRange(current_cell_faces); // face ids
    //-- Cell edge connectivity
    if (hasEdge())
      {
        cell_infos2.add(mesh()->edgeFamily()->itemKind()); // connected_family_id: edge family
        Integer current_cell_nb_edge = it->nbLocalEdge();
        cell_infos2.add(current_cell_nb_edge); // nb_connected_edges
        for (int edge_index = 0; edge_index < current_cell_nb_edge; ++edge_index) {
          Int64 first_node = current_cell_node_uids[it->localEdge(edge_index).beginNode()];
          Int64 second_node = current_cell_node_uids[it->localEdge(edge_index).endNode()];
          if (first_node > second_node) std::swap(first_node,second_node);// edge may be oriented negatively in the cell
          auto edge_it = edge_uid_map.find(std::make_pair(first_node,second_node));
          if (edge_it == edge_uid_map.end())
            ARCANE_FATAL("Do not find edge with nodes {0}-{1} in edge uid map. Exiting",
                         current_cell_node_uids[it->localEdge(edge_index).beginNode()],
                         current_cell_node_uids[it->localEdge(edge_index).endNode()]);
          cell_infos2.add(edge_it->second);
        }
      }
    cells_infos_index += current_cell_nb_node;
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillCellInfo2(Integer nb_cell,Int64ConstArrayView cells_infos,Int64Array& cell_infos2,
               Integer& nb_face, Int64Array& faces_infos, Int64Array& node_uids, bool allow_build_face)
{
  Integer cell_infos2_size_approx = 1 + 2 * (cells_infos.size()+ 2*nb_cell); // supposes as many faces as nodes...
  cell_infos2.reserve(cell_infos2_size_approx);
  // Fill infos
  ItemTypeMng* itm = m_item_type_mng;
  Integer cells_infos_index = 0;
  cell_infos2.add(2); // nb_connected_families (node and face)

  nb_face = 0;
  faces_infos.reserve(2*cells_infos.size());
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  NodeInFaceSet face_nodes_set;
  std::set<Int64> node_uid_set;
  // Fill face_infos2 from faces_infos
  for (Integer i_cell = 0; i_cell < nb_cell; ++i_cell) {

    Integer item_type_id = (Integer)cells_infos[cells_infos_index];
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell type
    cell_infos2.add(cells_infos[cells_infos_index++]); // cell uid
    cell_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id: node family

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    // Cell node connectivity
    Integer current_cell_nb_node = it->nbLocalNode();
    cell_infos2.add(current_cell_nb_node); // nb_connected_nodes
    Int64ConstArrayView current_cell_node_uids(current_cell_nb_node,&cells_infos[cells_infos_index]);
    cell_infos2.addRange(current_cell_node_uids); // node ids
    if (!allow_build_face) node_uid_set.insert(current_cell_node_uids.begin(),current_cell_node_uids.end()); // otherwise node ids will be computed in face info build
    // cell face connectivity
    cell_infos2.add(mesh()->faceFamily()->itemKind()); // connected_family_id: face family
    Integer current_cell_nb_face = it->nbLocalFace();
    cell_infos2.add(current_cell_nb_face); // nb_connected_faces
    for (Integer face_in_cell_index = 0; face_in_cell_index < current_cell_nb_face; ++face_in_cell_index)
      {
        ItemTypeInfo::LocalFace current_face = it->localFace(face_in_cell_index);
        Integer current_face_nb_node = current_face.nbNode();
        work_face_orig_nodes.resize(current_face_nb_node);
        work_face_sorted_nodes.resize(current_face_nb_node);
        Integer i = 0;
        for (Integer node_in_face_index = 0; node_in_face_index < current_face_nb_node; ++node_in_face_index)
          {
            work_face_orig_nodes[i++] = current_cell_node_uids[current_face.node(node_in_face_index)];
          }
        mesh_utils::reorderNodesOfFace(work_face_orig_nodes,work_face_sorted_nodes);
        // get face uid: we shouldn't have to guess, it should be given in arguments...
        Int64 current_face_uid = _findFaceUniqueId(work_face_sorted_nodes,face_nodes_set);
        if (current_face_uid == NULL_ITEM_ID) // face not already known add to face info
          {
            current_face_uid = m_face_uid_pool++;
            faces_infos.add(current_face.typeId()); // face info type
            faces_infos.add(current_face_uid); // face info uid
            faces_infos.addRange(work_face_sorted_nodes); // face info node uids
            _addFaceNodes(face_nodes_set,work_face_sorted_nodes,current_face_uid);
            ++nb_face;
          }
        cell_infos2.add(current_face_uid); // face ids
      }
    cells_infos_index += current_cell_nb_node;
  }
  if (! allow_build_face)
    {
      node_uids.resize(node_uid_set.size());
      Integer index = 0;
      for (Int64 node_uid : node_uid_set) node_uids[index++] = node_uid;
    }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info)
{
  _fillFaceRelationInfo(source_item_relation_data,target_item_dependencies_data,faces_info,true);
}


/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView faces_info)
{
  _fillFaceRelationInfo(source_item_relation_data,target_item_dependencies_data,faces_info,false);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView faces_info, bool is_source_relation_data_empty)
{
  // face_infos = [{face_type, face_uid, {face_node_uids}]
  Int64UniqueArray face_uids_and_types;
  face_uids_and_types.reserve(2*source_item_relation_data.nbItems());
  ItemTypeMng* itm = m_item_type_mng;
  for (Integer face_info_index = 0; face_info_index < faces_info.size();) {
    face_uids_and_types.add(faces_info[face_info_index+1]); // face_uid
    face_uids_and_types.add(faces_info[face_info_index]); // face_type
    Integer type_id = CheckedConvert::toInteger(face_uids_and_types.back());
    face_info_index += (2+itm->typeFromId(type_id)->nbLocalNode());// increment and skip info (first & second_node_uid)
  }
  _fillItemRelationInfo(source_item_relation_data,target_item_dependencies_data,face_uids_and_types, is_source_relation_data_empty);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView edges_info)
{
  _fillEdgeRelationInfo(source_item_relation_data,target_item_dependencies_data,edges_info,true);
}


/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView edges_info)
{
  _fillEdgeRelationInfo(source_item_relation_data,target_item_dependencies_data,edges_info,false);
}


/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView edges_info, bool is_source_relation_data_empty)
{
  // edge_infos = [{edge_uid, first_node_uid, second_node_uid}
  ARCANE_ASSERT((source_item_relation_data.nbItems()*3 == edges_info.size()),("source_item_relation_data and edges_info size incoherent. Exiting."));
  Int64UniqueArray edge_uids_and_types;
  edge_uids_and_types.reserve(2*source_item_relation_data.nbItems());
  for (Integer edge_info_index = 0; edge_info_index < edges_info.size();) {
    edge_uids_and_types.add(edges_info[edge_info_index++]); // edge_uid
    edge_uids_and_types.add(IT_Line2); // edge_type
    edge_info_index+=2;// skip info (first & second_node_uid)
  }
  _fillItemRelationInfo(source_item_relation_data,target_item_dependencies_data,edge_uids_and_types, is_source_relation_data_empty);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info)
{
  _fillNodeRelationInfo(source_item_relation_data,target_item_dependencies_data,nodes_info,true);
}


/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info)
{
  _fillNodeRelationInfo(source_item_relation_data,target_item_dependencies_data,nodes_info,false);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data, Int64ConstArrayView nodes_info, bool is_source_relation_data_empty){
  // nodes_info = [node_uids]
  ARCANE_ASSERT((source_item_relation_data.nbItems() == nodes_info.size()),("source_item_relation_data and nodes_info size incoherent. Exiting."));
  Int64UniqueArray node_uids_and_types(2*nodes_info.size());
  for (Integer node_index = 0; node_index < nodes_info.size(); ++node_index) {
    node_uids_and_types[2*node_index] = nodes_info[node_index];
    node_uids_and_types[2*node_index+1] = IT_Vertex;
  }
  _fillItemRelationInfo(source_item_relation_data,target_item_dependencies_data,node_uids_and_types, is_source_relation_data_empty);
}


/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillItemRelationInfo(ItemData& source_item_relation_data, const ItemData& target_item_dependencies_data,
                      Int64ConstArrayView source_item_uids_and_types,
                      bool is_source_item_relation_data_empty)
{
  ARCANE_ASSERT((source_item_relation_data.nbItems()*2 == source_item_uids_and_types.size()),
                ("source item number incoherent between source_item_relation_data and source_item_uids_and_types. Exiting."));
  // Fill an ItemData containing source to target relations (target to source relations are included in target_item_data)
  auto & source_relation_info = source_item_relation_data.itemInfos();
  const auto & target_dependencies_info = target_item_dependencies_data.itemInfos();
  auto source_family = source_item_relation_data.itemFamily();
  auto target_family = target_item_dependencies_data.itemFamily();
  if (! source_family || !target_family) return;
  std::map<Int64, Int64SharedArray> source_to_target_uids;
  Integer nb_families_connected_to_target = CheckedConvert::toInteger(target_dependencies_info[0]);
  Integer target_info_index = 1; // 0 is nb_connected_families

  // Fill map source to target traversing target_item_dependencies_data
  for (; target_info_index < target_dependencies_info.size();) {
    target_info_index++; // current target item_type
    Int64 target_item_uid = target_dependencies_info[target_info_index++];// current target item uid
    for (Integer family_connected_to_target = 0; family_connected_to_target < nb_families_connected_to_target; ++family_connected_to_target) {
      Int64 family_connected_to_target_kind = target_dependencies_info[target_info_index++];// current target item connected family kind
      if (family_connected_to_target_kind != source_family->itemKind()) {//this connection info does not concern source family. Skip
        Integer nb_non_read_values = CheckedConvert::toInteger(target_dependencies_info[target_info_index++]); // nb_connected_item on this other family (/= to source family)
        target_info_index+= nb_non_read_values;
        continue;
      }
      else {
        Int32 nb_source_item_connected_to_target_item = CheckedConvert::toInt32(target_dependencies_info[target_info_index++]);
        for (Integer source_item_index = 0; source_item_index < nb_source_item_connected_to_target_item; ++source_item_index) {
          source_to_target_uids[target_dependencies_info[source_item_index+target_info_index]].add(target_item_uid);
        }
        target_info_index += nb_source_item_connected_to_target_item;
      }
    }
  }
  // Fill or append source_item_relation_data traversing the map source_to_target_connectivity
  if (is_source_item_relation_data_empty) _initEmptyRelationInfo(source_relation_info, source_to_target_uids, source_item_uids_and_types,
                                                                 target_dependencies_info.size(), target_family);
  else _appendInitializedRelationInfo(source_relation_info, source_to_target_uids, source_item_uids_and_types,
                                      target_dependencies_info.size(), target_family);

}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_initEmptyRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                       Integer approx_relation_size,
                       IItemFamily const * target_family)
{
  if (! source_relation_info.empty()) source_relation_info.clear();
  source_relation_info.reserve(approx_relation_size); // clearly too much elements...but only available info. Do better ?
  source_relation_info.add(1);// nb_connected_families
  // Fill source_item_relation_data
  for (Integer source_item_uids_and_types_index = 0; source_item_uids_and_types_index < source_item_uids_and_types.size(); source_item_uids_and_types_index+=2) {
    Int64 source_item_uid = source_item_uids_and_types[source_item_uids_and_types_index];
    source_relation_info.add(source_item_uids_and_types[source_item_uids_and_types_index+1]); // item type
    source_relation_info.add(source_item_uid); // item uid
    source_relation_info.add(target_family->itemKind());// connected family kind
    auto & connected_items = source_to_target_uids[source_item_uid];
    source_relation_info.add(connected_items.size());// nb_connected_item
    source_relation_info.addRange(connected_items.view()); // connected_items uids
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_appendInitializedRelationInfo(Int64Array& source_relation_info, std::map<Int64, Int64SharedArray>& source_to_target_uids, Int64ConstArrayView source_item_uids_and_types,
                               Integer approx_relation_size,
                               IItemFamily const * target_family)
{
  // Create a working array to build the consolidated info. This array will be copied in to source_relation_info at the end.
  Int64UniqueArray source_relation_info_wrk_copy;
  source_relation_info_wrk_copy.reserve(source_relation_info.size()+approx_relation_size);
  std::set<Int64> treated_items;// To detect eventual source_items not already present in source_relation_info
  Integer source_relation_info_index = 0;
  Integer nb_connected_family = CheckedConvert::toInteger(source_relation_info[source_relation_info_index++]);
  source_relation_info_wrk_copy.add(nb_connected_family+1); // adding a new connected family
  for(; source_relation_info_index < source_relation_info.size() ;){
    source_relation_info_wrk_copy.add(source_relation_info[source_relation_info_index++]); // item type
    Int64 source_item_uid = source_relation_info[source_relation_info_index++]; // get item uid
    source_relation_info_wrk_copy.add(source_item_uid);
    treated_items.insert(source_item_uid);
    for (Integer connected_family_index = 0; connected_family_index < nb_connected_family; ++connected_family_index) {
      source_relation_info_wrk_copy.add(source_relation_info[source_relation_info_index++]); // family kind
      Integer nb_connected_elements = CheckedConvert::toInteger(source_relation_info[source_relation_info_index++]); // nb connected elements
      source_relation_info_wrk_copy.add(nb_connected_elements);
      source_relation_info_wrk_copy.addRange(source_relation_info.subConstView(source_relation_info_index,nb_connected_elements));
      source_relation_info_index += nb_connected_elements;
    }
    // Add new connection info
    source_relation_info_wrk_copy.add(target_family->itemKind()); // family kind
    const auto& connected_items = source_to_target_uids[source_item_uid]; // get connected items
    source_relation_info_wrk_copy.add(connected_items.size()); // nb connected elements
    source_relation_info_wrk_copy.addRange(connected_items);
  }
  // Find eventual source_items not already present in source_relation_info : todo check this can happen
  for (Integer source_item_uids_and_types_index = 0; source_item_uids_and_types_index < source_item_uids_and_types.size(); source_item_uids_and_types_index+=2) {
    Int64 source_item_uid = source_item_uids_and_types[source_item_uids_and_types_index];
    if (treated_items.find(source_item_uid) == treated_items.end()){
      source_relation_info_wrk_copy.add(source_item_uids_and_types[source_item_uids_and_types_index+1]); // item type
      source_relation_info_wrk_copy.add(source_item_uid); // item uid
      source_relation_info_wrk_copy.add(target_family->itemKind()); // family kind
      const auto& connected_items = source_to_target_uids[source_item_uid]; // get connected items
      source_relation_info_wrk_copy.add(connected_items.size()); // nb connected elements
      source_relation_info_wrk_copy.addRange(connected_items);
      // Padd this item info with family info : we added only one family, add void values for the other nb_connected_family families
      for (Integer connected_family_index = 0; connected_family_index < nb_connected_family; ++connected_family_index) {
        source_relation_info_wrk_copy.add(0); // family id (does not matter)
        source_relation_info_wrk_copy.add(0); // nb_connected_elements for this family
      }
    }
  }
  // Copy working array into destination array
  source_relation_info.resize(source_relation_info_wrk_copy.size());
  std::copy(source_relation_info_wrk_copy.begin(), source_relation_info_wrk_copy.end(), source_relation_info.begin());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DynamicMeshIncrementalBuilder::
_findFaceUniqueId(Int64ConstArrayView work_face_sorted_nodes, NodeInFaceSet& face_nodes_set)
{
  // check if first node exist
  Integer index = 0;
//  return _findFaceInFaceNodesSet(face_nodes_set,index,work_face_sorted_nodes,std::make_shared<NodeInFace>(NULL_ITEM_ID));
  return _findFaceInFaceNodesSet(face_nodes_set,index,work_face_sorted_nodes,NodeInFacePtr(NULL_ITEM_ID));
}

/*---------------------------------------------------------------------------*/

Int64 DynamicMeshIncrementalBuilder::
_findFaceInFaceNodesSet(const NodeInFaceSet& face_nodes_set,Integer index,Int64ConstArrayView face_nodes, NodeInFacePtr node)
{
  if (index < face_nodes.size())
    {
      auto found_node = std::find(face_nodes_set.begin(),face_nodes_set.end(),NodeInFacePtr(face_nodes[index]));
      if (found_node == face_nodes_set.end()) return node->faceUid();
      else return _findFaceInFaceNodesSet((*found_node)->nextNodeSet(),++index,face_nodes,*found_node);
    }
  else return node->faceUid();
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addFaceNodes(NodeInFaceSet& face_nodes_set, Int64ConstArrayView face_nodes, Int64 face_uid)
{
  Integer index = 0;
//  _addFaceInFaceNodesSet(face_nodes_set, index,face_nodes, std::make_shared<NodeInFace>(NULL_ITEM_ID), face_uid);
  _addFaceInFaceNodesSet(face_nodes_set, index,face_nodes, NodeInFacePtr(NULL_ITEM_ID), face_uid);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addFaceInFaceNodesSet(NodeInFaceSet& face_nodes_set,Integer index,Int64ConstArrayView face_nodes, NodeInFacePtr node, Int64 face_uid)
{
  if (index < face_nodes.size())
    {
//      auto next_node = std::make_shared<NodeInFace>(face_nodes[index]);
      auto next_node = _insertNode(face_nodes_set,face_nodes[index]);
//      debug() << "Add node " << face_nodes[index] << " in set " << &face_nodes_set;
//      debug() << "set size " << face_nodes_set.size();
      _addFaceInFaceNodesSet(next_node->nextNodeSet(),++index,face_nodes,next_node,face_uid);
//      else
//        _addFaceInFaceNodesSet((*insertion_return.first)->nextNodeSet(),++index,face_nodes,*insertion_return.first,face_uid); // pbe du const. Change from set to map ?
    }
  else
    {
      node->setFaceUid(face_uid);
//      debug() << "Set face uid " << face_uid << " for node " << node->m_uid;
    }
}

DynamicMeshIncrementalBuilder::NodeInFacePtr&
DynamicMeshIncrementalBuilder::_insertNode(NodeInFaceSet& face_nodes_set, Int64 inserted_node_uid)
{
  NodeInFacePtr new_node(inserted_node_uid);
  auto found_node = std::find(face_nodes_set.begin(),face_nodes_set.end(),new_node);
  if (found_node == face_nodes_set.end()) {
    face_nodes_set.push_back(std::move(new_node));
    return face_nodes_set.back();
  }
  else return *found_node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des items du maillage parent au en tant que maille au maillage actuel.
 *
 * \param items items à ajouter (vu depuis la maillage parent)
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 */
void DynamicMeshIncrementalBuilder::
addParentCells(const ItemVectorView & items)
{
  const Integer nb_cell = items.size();
  
  for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
    m_one_mesh_item_adder->addOneParentItem(items[i_cell], IK_Cell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Ajoute des mailles au maillage actuel.
 *
 * \param mesh_nb_cell nombre de mailles à ajouter
 * \param cells_infos infos sur les maillage (voir IMesh::allocateMesh())
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 * \param cells en retour, si non vide, contient les mailles créées.
 */
void DynamicMeshIncrementalBuilder::
addHChildrenCells(Cell hParent_cell,Integer nb_cell,Int64ConstArrayView cells_infos,
                  Integer sub_domain_id,Int32ArrayView cells,
                  bool allow_build_face)
{
  ItemTypeMng* itm = m_item_type_mng;
  CellFamily& cell_family = m_mesh->trueCellFamily();

  debug() << "[DynamicMeshIncrementalBuilder::addHChildrenCells] ADD CELLS mesh=" << m_mesh->name() << " nb_cell=" << nb_cell;
  Integer cells_infos_index = 0;
  bool add_to_cells = cells.size()!=0;
  if (add_to_cells && nb_cell!=cells.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'cells' has to have same size as number of cells");
  for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
    debug(Trace::Highest)<<"\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cell "<<i_cell<<"/"<<nb_cell;

    ItemTypeId item_type_id { (Int16)cells_infos[cells_infos_index] };
    debug(Trace::Highest)<<"\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cells_infos["<<cells_infos_index<<"]="<<cells_infos[cells_infos_index]<<", type_id="<<item_type_id;
    ++cells_infos_index;
	 
    Int64 cell_unique_id = cells_infos[cells_infos_index];
    debug(Trace::Highest)<<"\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]cells_infos["<<cells_infos_index<<"]="<<cells_infos[cells_infos_index]<<", uid="<<cell_unique_id;
    ++cells_infos_index;
    
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_cell_nb_node = it->nbLocalNode();
    debug(Trace::Highest) << "\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]DM ADD_CELL uid=" << cell_unique_id << " type_id=" << item_type_id
            << " nb_node=" << current_cell_nb_node;
    Int64ConstArrayView current_cell_nodes_uid(current_cell_nb_node,&cells_infos[cells_infos_index]);
    for( Integer i=0; i<current_cell_nb_node; ++i )
      debug(Trace::Highest) << "\t\t\t[DynamicMeshIncrementalBuilder::addHChildrenCells]DM NODE uid=" << current_cell_nodes_uid[i];

    ItemInternal* cell = m_one_mesh_item_adder->addOneCell(item_type_id,
                                                      cell_unique_id,
                                                      sub_domain_id,
                                                      current_cell_nodes_uid,
                                                      allow_build_face);

    cell_family._addParentCellToCell(cell,hParent_cell);
    if (add_to_cells)
      cells[i_cell] = cell->localId();
    cells_infos_index += current_cell_nb_node;
  }
  cell_family._addChildrenCellsToCell(hParent_cell,cells);

  debug() << "[DynamicMeshIncrementalBuilder::addHChildrenCells] done";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des items du maillage parent au en tant qu'items fantomes au maillage actuel.
 *
 * \param items items à ajouter (vu depuis la maillage parent)
 * \param sub_domain_id sous-domaine auquel les mailles appartiendront
 */
void DynamicMeshIncrementalBuilder::
addParentItems(const ItemVectorView & items, const eItemKind submesh_kind)
{
  const Integer nb_item = items.size();

  for( Integer i_item=0; i_item<nb_item; ++i_item ) {
    m_one_mesh_item_adder->addOneParentItem(items[i_item], submesh_kind, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des noeuds au maillage actuel.
 *
 * \param nodes_uid identifiant unique des noeuds à créer.
 * \param owner sous-domaine auquel les noeuds appartiendront.
 * \param nodes en retour, si non vide, contient les noeuds correspodants aux \a nodes_uid
 */
void DynamicMeshIncrementalBuilder::
addNodes(Int64ConstArrayView nodes_uid,Integer owner,Int32ArrayView nodes)
{
  Integer nb_node_to_add = nodes_uid.size();
  bool add_to_nodes = nodes.size()!=0;
  if (add_to_nodes && nb_node_to_add!=nodes.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'nodes' has to have same size as number of nodes");
  for( Integer i_node=0; i_node<nb_node_to_add; ++i_node ){
    ItemInternal* node = m_one_mesh_item_adder->addOneNode(nodes_uid[i_node],owner);
    if (add_to_nodes)
      nodes[i_node] = node->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des noeuds au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param nodes_uid identifiant unique des noeuds à créer.
 * \param owner sous-domaine auquel les noeuds appartiendront.
 * \param nodes en retour, si non vide, contient les noeuds correspodants aux \a nodes_uid
 */
void DynamicMeshIncrementalBuilder::
addNodes2(Int64ConstArrayView nodes_uid,Integer owner,Int32ArrayView nodes)
{
  Integer nb_node_to_add = nodes_uid.size();
  bool add_to_nodes = nodes.size()!=0;
  if (add_to_nodes && nb_node_to_add!=nodes.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'nodes' has to have same size as number of nodes");
  ItemDataList item_info_list;
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
      nb_node_to_add,2*nb_node_to_add+1,nodes,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),owner);
  Int64Array& node_infos2 = node_data.itemInfos();
  // Copy node_info : to decrease the footprint of the copy, use references instead of int64
  _fillNodeNewInfo(nodes_uid,node_infos2);

  addItems(item_info_list);
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillNodeNewInfo(Int64ConstArrayView node_uids,Int64ArrayView node_infos2)
{
  node_infos2[0] = 0; // nb_connected_families
  Integer index = 0;
  for (auto node_uid : node_uids)
    {
      node_infos2[2*index+1] = IT_Vertex; // type
      node_infos2[2*index+2] = node_uid; // id
      index++;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajout générique d'items d'un ensemble de famille pour lesquelles on fournit un ItemData.
 *
 * L'objet \a ItemDataList est une map <family_index,ItemData> où family_index est pris égal à l'item_kind de la famille
 * et où ItemData aggrège les informations id/connectivités des items Le tableau item_infos (ItemData::itemInfos()) à la structure suivante :
 * item_infos[0]   = nb_connected_families // Only constitutive (owning) connections.
 * item_infos[i]   = first_item_type
 * item_infos[i+1] = first_item_uid
 * item_infos[i+2] = first_family_id
 * item_infos[i+3] = nb_connected_items_in_first_family
 * item_infos[i+4...i+n] = first_family items uids
 * item_infos[i+n+1] = second_family_id
 * item_infos[i+n+1...i+m] = second_family items uids
 * item_infos[i+m+1] = second_item_uid
 * ...idem first item
 * La méthode parcours le graphe des connectivités du maillage pour créer les items de toute les familles
 * La méthode addFamilyItems(ItemInfo&) crée les items d'une famille donnée
 */

void DynamicMeshIncrementalBuilder::
addItems(ItemDataList& item_info_list)
{
  // Add items : item construction done from leaves to root
  _addItemsOrRelations(item_info_list,IItemFamilyNetwork::InverseTopologicalOrder);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addItems(ItemDataList& item_info_list, ItemDataList& item_relation_info_list)
{
  if (item_info_list.size() == 0 || item_relation_info_list.size() == 0) return;
  addItems(item_info_list);
  addRelations(item_relation_info_list);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addRelations(ItemDataList& item_relation_list)
{
  // Add relations : items are already added. Do it from root to leaves
  _addItemsOrRelations(item_relation_list,IItemFamilyNetwork::TopologicalOrder);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_addItemsOrRelations(ItemDataList& info_list, IItemFamilyNetwork::eSchedulingOrder family_graph_traversal_order)
{
  // Add relations : items are already added. Do it from root to leaves
  if (info_list.size() == 0) return;
    m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
    ItemData i_infos = info_list[family->itemKind()];
    this->addFamilyItems(i_infos);
    },
    family_graph_traversal_order); // item construction done from leaves to root
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajout générique d'items d'une famille, décrite par son ItemInfo
 *
 */

void DynamicMeshIncrementalBuilder::
addFamilyItems(ItemData& item_data)
{
  if (item_data.nbItems() == 0) return;
  bool add_to_items = item_data.itemLids().size()!=0;
  if (add_to_items && item_data.nbItems()!= item_data.itemLids().size())
     throw ArgumentException(A_FUNCINFO,
                             "return array containing item lids has to have be of size number of items");
  // Prepare info and call OneMeshItemAdder->addOneItem
  Int64 item_uid;
  Int64ConstArrayView item_infos = item_data.itemInfos();
  Integer nb_connected_family = CheckedConvert::toInteger(item_infos[0]);
  if(nb_connected_family == 0)
    return ;
  Int64UniqueArray connectivity_info;
  Integer nb_item_info = 0;
  Integer item_index = 0;
  for (Integer info_index = 1; info_index < item_data.itemInfos().size();){
    ItemTypeId item_type_id { (Int16)item_infos[info_index++] }; // item_type
    item_uid = item_infos[info_index++]; // item_uid
    Integer item_info_begin_index = info_index;
    for (Integer connected_family_index = 0;connected_family_index < nb_connected_family; ++connected_family_index) {
      Integer current_index = CheckedConvert::toInteger(item_infos[info_index+1]);
      Integer nb_item_info_increment = 2 + current_index;
      nb_item_info += nb_item_info_increment; // family_id , nb_connected_elements
      info_index+= nb_item_info_increment; // pointing on next family id
    }
    ItemInternal* item = m_one_mesh_item_adder->addOneItem2(item_data.itemFamily(),
                                                            item_data.itemFamilyModifier(),
                                                            item_type_id,
                                                            item_uid,
                                                            item_data.itemOwners()[item_index],
                                                            item_data.subDomainId(),
                                                            nb_connected_family,
                                                            item_infos.subView(item_info_begin_index,nb_item_info));
    if (add_to_items)
      item_data.itemLids()[item_index++] = item->localId();
    nb_item_info = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute une face.
 *
 * Ajoute une face en fournissant l'unique_id à utiliser et les unique_ids
 * des noeuds à connecter.
 */
ItemInternal *DynamicMeshIncrementalBuilder::
addFace(Int64 a_face_uid, Int64ConstArrayView a_node_list, Integer a_type)
{
  return m_one_mesh_item_adder->addOneFace(a_face_uid,a_node_list,a_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au maillage actuel.
 *
 * \param nb_face nombre de faces à ajouter
 * \param faces_infos infos sur les faces (voir IMesh::allocateCells())
 * \param sub_domain_id sous-domaine auquel les faces appartiendront
 * \param faces en retour, si non vide, contient les faces créées.
 */
void DynamicMeshIncrementalBuilder::
addFaces(Integer nb_face,Int64ConstArrayView faces_infos,
         Integer sub_domain_id,Int32ArrayView faces)
{
  ItemTypeMng* itm = m_item_type_mng;
  
  bool add_to_faces = faces.size()!=0;
  if (add_to_faces && nb_face!=faces.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'faces' has to have same size as number of faces");

  Integer faces_infos_index = 0;
  for(Integer i_face=0; i_face<nb_face; ++i_face ){

    ItemTypeId item_type_id { (Int16)faces_infos[faces_infos_index] };
    ++faces_infos_index;
    Int64 face_unique_id = faces_infos[faces_infos_index];
    ++faces_infos_index;
    
    ItemTypeInfo* it = itm->typeFromId(item_type_id);

    Integer current_face_nb_node = it->nbLocalNode();
    Int64ConstArrayView nodes_uid(current_face_nb_node,&faces_infos[faces_infos_index]);
    faces_infos_index += current_face_nb_node;

    ItemInternal* face = m_one_mesh_item_adder->addOneFace(item_type_id, face_unique_id, sub_domain_id, nodes_uid);
    
    if (add_to_faces)
      faces[i_face] = face->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param nb_face nombre de faces à ajouter
 * \param faces_infos infos sur les faces (voir IMesh::allocateCells())
 * \param sub_domain_id sous-domaine auquel les faces appartiendront
 * \param faces en retour, si non vide, contient les faces créées.
 */
void DynamicMeshIncrementalBuilder::
addFaces2(Integer nb_face,Int64ConstArrayView faces_infos,
          Integer sub_domain_id,Int32ArrayView faces)
{
  bool add_to_faces = faces.size()!=0;
  if (add_to_faces && nb_face!=faces.size())
    throw ArgumentException(A_FUNCINFO,
                            "return array 'faces' has to have same size as number of faces");
  // Prepare face infos 2
  Integer face_infos2_size = faces_infos.size()+ 1 + 2 * nb_face; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
  ItemDataList item_info_list;
  ItemData& face_data = item_info_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),
                                                  nb_face,face_infos2_size,faces,&mesh()->trueFaceFamily(),&mesh()->trueFaceFamily(),sub_domain_id);

  Int64ArrayView face_infos2 = face_data.itemInfos().view();
  Int64UniqueArray node_uids;
  Int64UniqueArray edge_uids;

  _fillFaceInfo2(nb_face,faces_infos,face_infos2,node_uids);

  // Prepare edge data if needed
//  if (m_has_edge) {
//    Integer nb_edge_to_add = edge_uids.size();
//    Int32UniqueArray edge_lids(nb_edge_to_add);
//    ItemData& edge_data = item_info_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),
//          nb_edge_to_add,2*nb_edge_to_add+1,edge_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
//    Int64Array& edge_infos2 = edge_data.itemInfos();
//    _fillEdgeInfo2(nb_edge_to_add,edge_uids,edge_infos2);
//  }

  // Prepare node data (mutualize)
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
        nb_node_to_add,2*nb_node_to_add+1,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids,node_infos2);

  // Launch node and face creation
  addItems(item_info_list);
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addFaces3(Integer nb_face,Int64ConstArrayView faces_infos,
          Integer sub_domain_id,Int32ArrayView faces)
{
  bool add_to_faces = faces.size()!=0;
  if (add_to_faces && nb_face!=faces.size())
    throw ArgumentException(A_FUNCINFO,
        "return array 'faces' has to have same size as number of faces");
  ItemDataList item_data_list;
  ItemDataList item_relation_data_list;
  // item info_size cannot be guessed
  Int64UniqueArray edges_infos;
  Int64UniqueArray nodes_infos;
  Int64UniqueArray node_uids;
  Int32UniqueArray edge_lids;
  Int32UniqueArray node_lids;
  Integer nb_edge = 0;
  Integer nb_node = 0;
  std::map<std::pair<Int64,Int64>, Int64> edge_uid_map;

  // Fill classical info
  _fillEdgeInfo(nb_edge, nb_face, edges_infos, faces_infos, edge_uid_map);
  _fillNodeInfo(nb_node, nb_face, nodes_infos, faces_infos);

  // Face info New
  ItemData& face_data = item_data_list.itemData(Integer(m_mesh->faceFamily()->itemKind()),nb_face,0,faces,&mesh()->trueFaceFamily(),&mesh()->trueFaceFamily(),sub_domain_id);
  _fillFaceNewInfoNew(nb_face,faces_infos,face_data.itemInfos(),edge_uid_map);

  // Edge info New
  if (hasEdge()) {
    Integer edge_info_size = 1 + nb_edge*6;//New info = Nb_connected_family + for each edge : type, uid, connected_family_kind, nb_node_connected, first_node_uid, second_node_uid
    ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),nb_edge,edge_info_size,edge_lids,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);
    edge_lids.resize(item_data_list[m_mesh->edgeFamily()->itemKind()].nbItems());
    _fillEdgeNewInfoNew(nb_edge,edges_infos,edge_data.itemInfos()); // reprendre fillEdgeInfo et voir l'appel depuis add Edges + attention ordre des noeuds,
    ItemData& edge_relation_data = item_relation_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),nb_edge,edge_info_size,edge_lids,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);
    _initEdgeRelationInfo(edge_relation_data,face_data, edges_infos); // edge-face relation
    }

  // Node info New
  Integer node_info_size = 1 + nb_node*2;//New info = Nb_connected_family + for each node : type, uid (no connected family)
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,node_info_size,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  node_lids.resize(item_data_list[m_mesh->nodeFamily()->itemKind()].nbItems());
  _fillNodeNewInfo(nodes_infos,node_data.itemInfos());
  ItemData& node_relation_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,node_info_size,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  _initNodeRelationInfo(node_relation_data,face_data, nodes_infos); // node-face relation


  // Launch node, face and edges (not compulsory) creation
  addItems(item_data_list, item_relation_data_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceNewInfoNew(Integer nb_face,Int64ConstArrayView faces_infos,Int64Array& face_infos2, const std::map<std::pair<Int64,Int64>, Int64>& edge_uid_map)
{
  ItemTypeMng* itm = m_item_type_mng;
  Integer nb_connected_family = hasEdge() ? 2 : 1;
  Integer face_infos2_size_approx = 1 + 2 * (faces_infos.size()+ 2*nb_face); // supposes as many faces as edges...
  face_infos2.reserve(face_infos2_size_approx);
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  Integer faces_infos_index = 0;
//  Integer face_infos2_index = 0;
  std::set<Int64> node_uids_set;
  face_infos2.add(nb_connected_family); // nb_connected_families nodes +/- Edges !!
  // Fill face_infos2 from faces_infos
  for(Integer i_face=0; i_face<nb_face; ++i_face ){
    Integer current_face_type = (Integer)faces_infos[faces_infos_index++];
    Integer current_face_uid = (Integer)faces_infos[faces_infos_index++];
    face_infos2.add(current_face_type); // type
    face_infos2.add(current_face_uid); // uid
    //--Node connectivity
    face_infos2.add(mesh()->nodeFamily()->itemKind()); // connected_family_id
    ItemTypeInfo* it = itm->typeFromId(current_face_type);
    Integer current_face_nb_node = it->nbLocalNode();
    face_infos2.add(current_face_nb_node); // nb_connected_nodes
    // Nodes are not necessarily sorted (if caller is addFaces...)
    Int64ConstArrayView current_face_node_uids(current_face_nb_node,&faces_infos[faces_infos_index]);
    work_face_orig_nodes.resize(current_face_nb_node);
    work_face_sorted_nodes.resize(current_face_nb_node);
    work_face_orig_nodes.copy(current_face_node_uids);
    mesh_utils::reorderNodesOfFace(work_face_orig_nodes,work_face_sorted_nodes);
    face_infos2.addRange(work_face_sorted_nodes); // connected node uids
    faces_infos_index += current_face_nb_node;
    //--Edge Connectivity
    if (hasEdge()){
      face_infos2.add(mesh()->edgeFamily()->itemKind());
      face_infos2.add(it->nbLocalEdge()); // nb_connected_edges
      for (int edge_index = 0; edge_index < it->nbLocalEdge(); ++edge_index) {
        Int64 first_node  = work_face_sorted_nodes[it->localEdge(edge_index).beginNode()];
        Int64 second_node = work_face_sorted_nodes[it->localEdge(edge_index).endNode()];
        if (first_node > second_node) std::swap(first_node,second_node); // edge may be oriented negatively in the face
        auto edge_it = edge_uid_map.find(std::make_pair(first_node,second_node));
        if (edge_it == edge_uid_map.end())
          ARCANE_FATAL("Do not find edge with nodes {0}-{1} in edge uid map. Exiting",
                       work_face_sorted_nodes[it->localEdge(edge_index).beginNode()],
                       work_face_sorted_nodes[it->localEdge(edge_index).endNode()]);
        face_infos2.add(edge_it->second); // connected edge uid
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillFaceInfo2(Integer nb_face,Int64ConstArrayView faces_infos,Int64ArrayView face_infos2, Int64Array& node_uids)
{
  ItemTypeMng* itm = m_item_type_mng;
  Integer faces_infos_index = 0;
  Integer face_infos2_index = 0;
  face_infos2[face_infos2_index++] = 1; // nb_connected_families (node only)
  Int64UniqueArray work_face_sorted_nodes;
  Int64UniqueArray work_face_orig_nodes;
  std::set<Int64> node_uids_set;
  // Fill face_infos2 from faces_infos
  for(Integer i_face=0; i_face<nb_face; ++i_face ){

    Integer item_type_id = (Integer)faces_infos[faces_infos_index];
    face_infos2[face_infos2_index++] = faces_infos[faces_infos_index++]; // type
    face_infos2[face_infos2_index++] = faces_infos[faces_infos_index++]; // uid
    face_infos2[face_infos2_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id

    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    Integer current_face_nb_node = it->nbLocalNode();
    face_infos2[face_infos2_index++] = current_face_nb_node; // nb_connected_items
    Int64ConstArrayView current_face_node_uids(current_face_nb_node,&faces_infos[faces_infos_index]);
    work_face_orig_nodes.resize(current_face_nb_node);
    work_face_sorted_nodes.resize(current_face_nb_node);
    work_face_orig_nodes.copy(current_face_node_uids);
    mesh_utils::reorderNodesOfFace(work_face_orig_nodes,work_face_sorted_nodes);
    face_infos2.subView(face_infos2_index,current_face_nb_node).copy(work_face_sorted_nodes); // connected node uids
    faces_infos_index += current_face_nb_node;
    face_infos2_index += current_face_nb_node;
    node_uids_set.insert(current_face_node_uids.begin(),current_face_node_uids.end());
  }
  node_uids.resize(node_uids_set.size());
  Integer index = 0;
  for (Int64 node_uid : node_uids_set) node_uids[index++] = node_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des arêtes au maillage actuel.
 *
 * \param nb_face nombre de faces à ajouter
 * \param edge_infos infos sur les arêtes (voir IMesh::allocateCells() sans identifiant de type)
 * \param sub_domain_id sous-domaine auquel les arêtes appartiendront
 * \param edges en retour, si non vide, contient les arêtes créées.
 */
void DynamicMeshIncrementalBuilder::
addEdges(Integer nb_edge,Int64ConstArrayView edge_infos,
         Integer sub_domain_id,Int32ArrayView edges)
{
  bool add_to_edges = edges.size()!=0;
    if (add_to_edges && nb_edge!=edges.size())
      throw ArgumentException(A_FUNCINFO,
                              "return array 'edges' has to have same size as number of edges");

    Integer edge_info_index = 0;
    for(Integer i_edge=0; i_edge<nb_edge; ++i_edge ){

      Int64 edge_unique_id = edge_infos[edge_info_index];
      ++edge_info_index;

      Int64ConstArrayView nodes_uid(2,&edge_infos[edge_info_index]);
      edge_info_index += 2;

      ItemInternal* edge = m_one_mesh_item_adder->addOneEdge(edge_unique_id, sub_domain_id, nodes_uid);

      if (add_to_edges)
        edges[i_edge] = edge->localId();
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des arêtes au maillage actuel. Utilise l'ajout d'item générique basé sur dépendances entre familles.
 *
 * \param nb_face nombre de faces à ajouter
 * \param edge_infos infos sur les arêtes (voir IMesh::allocateCells() sans identifiant de type)
 * \param sub_domain_id sous-domaine auquel les arêtes appartiendront
 * \param edges en retour, si non vide, contient les arêtes créées.
 */
void DynamicMeshIncrementalBuilder::
addEdges2(Integer nb_edge,Int64ConstArrayView edge_infos,
         Integer sub_domain_id,Int32ArrayView edges)
{
  bool add_to_edges = edges.size()!=0;
  if (add_to_edges && nb_edge!=edges.size())
    throw ArgumentException(A_FUNCINFO,
                              "return array 'edges' has to have same size as number of edges");

  // Prepare face infos 2
  Integer edge_infos2_size = edge_infos.size()+ 1 + 2 * nb_edge; // additive info nb_connected_family + nb_face * (family_id + nb_connected_element)
  ItemDataList item_info_list;
  ItemData& edge_data = item_info_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),
      nb_edge,edge_infos2_size,edges,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);

  Int64ArrayView edge_infos2 = edge_data.itemInfos().view();
  Int64UniqueArray node_uids;

  _fillEdgeInfo2(nb_edge,edge_infos,edge_infos2,node_uids);

  // Prepare node data (mutualize)
  Integer nb_node_to_add = node_uids.size();
  Int32UniqueArray node_lids(nb_node_to_add);
  ItemData& node_data = item_info_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),
      nb_node_to_add,2*nb_node_to_add+1,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  Int64Array& node_infos2 = node_data.itemInfos();
  _fillNodeNewInfo(node_uids,node_infos2);

  // Launch node and face creation
  addItems(item_info_list);
}

void DynamicMeshIncrementalBuilder::
addEdges3(Integer nb_edge,Int64ConstArrayView edges_infos,
         Integer sub_domain_id,Int32ArrayView edges)
{
  bool add_to_edges = edges.size()!=0;
  if (add_to_edges && nb_edge!=edges.size())
    throw ArgumentException(A_FUNCINFO,
        "return array 'edges' has to have same size as number of edges");
  ItemDataList item_data_list;
  ItemDataList item_relation_data_list;
  // item info_size cannot be guessed
  Int64UniqueArray nodes_infos;
  Int32UniqueArray node_lids;
  Integer nb_node = 0;

  // Fill classical info
  _fillNodeInfoFromEdge(nb_node, nb_edge, nodes_infos, edges_infos);

  // Edge new info : edge data stores edge uids and dependency connectivities
  ItemData& edge_data = item_data_list.itemData(Integer(m_mesh->edgeFamily()->itemKind()),nb_edge,0,edges,&mesh()->trueEdgeFamily(),&mesh()->trueEdgeFamily(),sub_domain_id);
  _fillEdgeNewInfoNew(nb_edge,edges_infos,edge_data.itemInfos());


  // Node new info & relationsm
  ItemData& node_data = item_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,0,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  node_lids.resize(nb_node);
  _fillNodeNewInfo(nodes_infos,node_data.itemInfos());
  ItemData& node_relations_data = item_relation_data_list.itemData(Integer(m_mesh->nodeFamily()->itemKind()),nb_node,0,node_lids,&mesh()->trueNodeFamily(),&mesh()->trueNodeFamily(),sub_domain_id);
  _initNodeRelationInfo(node_relations_data,edge_data, nodes_infos); // node-edge relation


  // Launch node and edges creation
  addItems(item_data_list, item_relation_data_list);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_fillEdgeInfo2(Integer nb_edge,Int64ConstArrayView edges_infos,Int64ArrayView edge_infos2, Int64Array& node_uids)
{
  Integer edges_infos_index = 0;
  Integer edge_infos2_index = 0;
  std::set<Int64> node_uids_set;

  // Fill edge_infos2 from edges_infos
  edge_infos2[edge_infos2_index++] = 1; // nb_connected_families (node only)

  for(Integer i_edge=0; i_edge<nb_edge; ++i_edge ){
    edge_infos2[edge_infos2_index++] = IT_Line2; // type
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index++]; // uid
    edge_infos2[edge_infos2_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id
    edge_infos2[edge_infos2_index++] = 2; // nb_connected_items Attention : ordre des noeuds !! TODO
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index]; // first connected node uid
    node_uids_set.insert(edges_infos[edges_infos_index++]);
    edge_infos2[edge_infos2_index++] = edges_infos[edges_infos_index]; // second connected node uid
    node_uids_set.insert(edges_infos[edges_infos_index++]);
  }
  node_uids.resize(node_uids_set.size());
  Integer index = 0;
  for (Int64 node_uid : node_uids_set) node_uids[index++] = node_uid;
}

void DynamicMeshIncrementalBuilder::
_fillEdgeNewInfoNew(Integer nb_edge,Int64ConstArrayView edges_infos,Int64ArrayView edges_new_infos)
{
  Integer edges_infos_index = 0;
  Integer edges_new_infos_index = 0;

  // Fill edge_infos2 from edges_infos
  edges_new_infos[edges_new_infos_index++] = 1; // nb_connected_families (node only)

  for(Integer i_edge=0; i_edge<nb_edge; ++i_edge ){
    edges_new_infos[edges_new_infos_index++] = IT_Line2; // type
    edges_new_infos[edges_new_infos_index++] = edges_infos[edges_infos_index++]; // uid
    edges_new_infos[edges_new_infos_index++] = mesh()->nodeFamily()->itemKind(); // connected_family_id
    edges_new_infos[edges_new_infos_index++] = 2; // nb_connected_items
    Int64 first_node  = edges_infos[edges_infos_index++];
    Int64 second_node = edges_infos[edges_infos_index++];
    if (first_node > second_node) std::swap(first_node,second_node);
    edges_new_infos[edges_new_infos_index++] = first_node; // first connected node uid
    edges_new_infos[edges_new_infos_index++] = second_node; // second connected node uid
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
readFromDump()
{
  // Recalcul le max uniqueId() des faces pour savoir quelle valeur
  // utiliser en cas de creation de face.
  ItemInternalMap& faces_map = m_mesh->facesMap();
  Int64 max_uid = -1;
  faces_map.eachItem([&](Item face) {
    if (face.uniqueId() > max_uid)
      max_uid = face.uniqueId();
  });
  m_one_mesh_item_adder->setNextFaceUid(max_uid + 1);

  if (m_has_edge) {
    ItemInternalMap& edges_map = m_mesh->edgesMap();
    Int64 max_uid = -1;
    edges_map.eachItem([&](Item edge) {
      if (edge.uniqueId() > max_uid)
        max_uid = edge.uniqueId();
    });
    m_one_mesh_item_adder->setNextEdgeUid(max_uid + 1);
  }

  info(5) << "NEXT FACE UID mesh=" << m_mesh->name() << " Next=" << m_one_mesh_item_adder->nextFaceUid();
  info(5) << "NEXT EDGE UID mesh=" << m_mesh->name() << " Next=" << m_one_mesh_item_adder->nextEdgeUid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
printInfos()
{
  info() << "-- Mesh information (Arcane2):";
  info() << "Number of nodes   " << m_one_mesh_item_adder->nbNode();
  info() << "Number of edges   " << m_one_mesh_item_adder->nbEdge();
  info() << "Number of faces   " << m_one_mesh_item_adder->nbFace();
  info() << "Number of cells   " << m_one_mesh_item_adder->nbCell();
  info() << "-- Graph information (Arcane2):";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
printStats(Int32 level)
{ 
  info(level) << "-- -- Statistics";
  info(level) << "Number of nodes after addition     &: " << m_one_mesh_item_adder->nbNode()
              << " hashmap=" << m_mesh->nodesMap().count();
  info(level) << "Number of edges after addition     : " << m_one_mesh_item_adder->nbEdge()
              << " hashmap=" << m_mesh->edgesMap().count();
  info(level) << "Number of faces after addition     : " << m_one_mesh_item_adder->nbFace()
              << " hashmap=" << m_mesh->facesMap().count();
  info(level) << "Number of cells after addition     : " << m_one_mesh_item_adder->nbCell()
              << " hashmap=" << m_mesh->cellsMap().count();
  info(level) << "--";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque face.

  L'algorithme doit fonctionner de manière à donner la même numérotation des
  faces séquentiel et en parallèle quel que soit le découpage, afin
  de faciliter le débogage.

  Le principe de numérotation est le suivant: on parcours les mailles dans
  l'ordre croissant de leur unique_id et pour chaque maille on parcourt
  la liste des faces. Une face est numérotée si et seulement si elle a
  pour backCell() la maille courante ou si elle a pour frontCell() la
  maille courante mais qu'elle est frontière (nbCell()==1). Quand on
  numérote les faces de la maille courante, on numérote d'abord les faces
  dont elle est la backCell(), puis les faces dont elle est la frontCell().

  Ce petit détail s'explique pour simplifier la détermination de la
  numérotation dans le cas parallèle (todo:expliquer pourquoi...)

  \warning Cette méthode ne doit être appelée que lors de la création
  du maillage initial.
*/
void DynamicMeshIncrementalBuilder::
computeFacesUniqueIds()
{
  if (!m_face_unique_id_builder)
    m_face_unique_id_builder = new FaceUniqueIdBuilder(this);
  m_face_unique_id_builder->computeFacesUniqueIds();

  if (m_has_edge) {
    if (!m_edge_unique_id_builder)
      m_edge_unique_id_builder = new EdgeUniqueIdBuilder(this);
    m_edge_unique_id_builder->computeEdgesUniqueIds();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
addGhostLayers(bool is_allocate)
{
  debug() << "Add one ghost layer";
  if (!m_ghost_layer_builder)
    m_ghost_layer_builder = new GhostLayerBuilder(this);
  m_ghost_layer_builder->addGhostLayers(is_allocate);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR
void DynamicMeshIncrementalBuilder::
addGhostChildFromParent(Array<Int64>& ghost_cell_to_refine)
{
  debug() << "Add one ghost layer";
  if (!m_ghost_layer_builder)
    m_ghost_layer_builder = new GhostLayerBuilder(this);
  m_ghost_layer_builder->addGhostChildFromParent2(ghost_cell_to_refine);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
_removeNeedRemoveMarkedItems(ItemInternalMap& map, UniqueArray<Int32>& items_local_id)
{
  // Suppression des liaisons
  SharedArray<ItemInternal*> items_to_remove;
  items_to_remove.reserve(1000);

  map.eachItem([&](Item item) {
    impl::MutableItemBase mb_item = item.mutableItemBase();
    Integer f = mb_item.flags();
    if (f & ItemFlags::II_NeedRemove){
      f &= ~ItemFlags::II_NeedRemove;
      mb_item.setFlags(f);
      items_local_id.add(item.localId());
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Supprime les items fantômes.
 *
 Supprime tous les items dont le propriétaire n'est pas le sous-domaine actuel
 et dont aucun éléments internes n'appartient à ce sous-domaine.
 Les items internes qui ne sont plus connectés à des items sont
 eux aussi détruits
*/
void DynamicMeshIncrementalBuilder::
removeNeedRemoveMarkedItems()
{ 
  if(!m_mesh->useMeshItemFamilyDependencies())
  {

    IItemFamily* links_family = m_mesh->findItemFamily(IK_DoF, GraphDoFs::linkFamilyName(), false, false);
    if(links_family)
    {
      info() << "Remove Items for family : "<<links_family->name() ;
      links_family->removeNeedRemoveMarkedItems() ;
    }

    for( IItemFamily* family : m_mesh->itemFamilies() )
    {
      if (family->itemKind()!=IK_DoF || family->name()==GraphDoFs::linkFamilyName())
        continue;
      info() << "Remove Items for family : "<<family->name() ;
      family->removeNeedRemoveMarkedItems() ;
    }
    // Supression des particules de la famille Particle

    for( IItemFamilyCollection::Enumerator i(m_mesh->itemFamilies()); ++i; ){
      IItemFamily* family = *i;
      if (family->itemKind()==IK_Particle){
        ParticleFamily* particle_family = dynamic_cast<ParticleFamily*>(family) ;
        ARCANE_CHECK_POINTER(particle_family);
        if(particle_family && particle_family->getEnableGhostItems()==true){
          UniqueArray<Integer> lids_to_remove ;
          lids_to_remove.reserve(1000);

          ItemInternalMap& particle_map = particle_family->itemsMap();
          particle_map.eachItem([&](Item item) {
            impl::MutableItemBase mb_item = item.mutableItemBase();
            Integer f = mb_item.flags();
            if (f & ItemFlags::II_NeedRemove){
              f &= ~ItemFlags::II_NeedRemove;
              mb_item.setFlags(f);
              lids_to_remove.add(item.localId());
            }
          });

          info() << "Number of particles of family "<<family->name()<<" to remove: " << lids_to_remove.size();
          if(lids_to_remove.size()>0)
            particle_family->removeParticles(lids_to_remove) ;
        }
      }
    }

    // Suppression des mailles
    CellFamily& cell_family = m_mesh->trueCellFamily();
    UniqueArray<Int32> cells_to_remove;
    cells_to_remove.reserve(1000);
    _removeNeedRemoveMarkedItems(cell_family.itemsMap(), cells_to_remove);

    info() << "Number of cells to remove: " << cells_to_remove.size();

#ifdef ARCANE_DEBUG_DYNAMIC_MESH
    for( Integer i=0, is=cells_to_remove.size(); i<is; ++i )
      info() << "remove cell with uid=" << cells_to_remove[i]->uniqueId()
             << ",lid=" << cells_to_remove[i]->localId()
             << ",owner=" << cells_to_remove[i]->owner();
#endif /* ARCANE_DEBUG_DYNAMIC_MESH */

    cell_family.removeCells(cells_to_remove);
  }
  // With ItemFamilyNetwork
  else
  {
    // handle families already in the network
    m_mesh->itemFamilyNetwork()->schedule([](IItemFamily* family){
                                            family->removeNeedRemoveMarkedItems();
                                            // Remove items from family graph leaves to root
                                          }, IItemFamilyNetwork::InverseTopologicalOrder);
    for (auto family : m_mesh->itemFamilies()) {
      // handle remaining families (particles, links, dual nodes & dof).
      // Will have no effect on the families already handled in the graph (since NeedRemove flag is reset)
      family->removeNeedRemoveMarkedItems();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
setConnectivity(const Integer connectivity)
{
  if (connectivity == 0)
    ARCANE_FATAL("Undefined connectivity !");
  if (m_connectivity != 0)
    ARCANE_FATAL("Connectivity already set: cannot redefine it");
  m_connectivity = connectivity;
  m_has_edge = Connectivity::hasConnectivity(connectivity,Connectivity::CT_HasEdge);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshIncrementalBuilder::
resetAfterDeallocate()
{
  m_face_uid_pool = 0;
  m_edge_uid_pool = 0;
  m_one_mesh_item_adder->resetAfterDeallocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
