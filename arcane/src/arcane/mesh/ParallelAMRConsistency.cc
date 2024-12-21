// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*  ParallelAMRConsistency.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Consistence parallèle des uid des noeuds/faces dans le cas AMR            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Collection.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/MapCoordToUid.h"
#include "arcane/mesh/ParallelAMRConsistency.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Item.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemCompare.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/GeometricUtilities.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMeshUtilities.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR

#ifdef ACTIVATE_PERF_COUNTER
const std::string ParallelAMRConsistency::PerfCounter::m_names[ParallelAMRConsistency::PerfCounter::NbCounters] =
  {
    "INIT",
    "COMPUTE",
    "GATHERFACE",
    "UPDATE",
    "REHASH",
    "ENDUPDATE"
  };
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelAMRConsistency::
ParallelAMRConsistency(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_nodes_coord(m_mesh->toPrimaryMesh()->nodesCoordinates())
, m_nodes_info(1000, true)
, m_active_nodes(1000, true)
, m_active_faces(5000, true)
, m_active_faces2(5000, true)
, m_is_updated(false)
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelAMRConsistency::
init()
{
  CHECKPERF( m_perf_counter.start(PerfCounter::INIT) )
  // Marque les noeuds sur la frontière
  Integer nb_active_face = static_cast<Integer> (m_mesh->nbFace() * 0.2); // 20% de faces shared (sur estimé)
  m_shared_face_uids.reserve(nb_active_face) ;
  m_shared_face_uids.clear() ;
  m_connected_shared_face_uids.reserve(nb_active_face) ;
  m_connected_shared_face_uids.clear() ;
  Integer sid = m_mesh->parallelMng()->commRank();
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Face face = *iface;
    ItemUniqueId face_uid = face.uniqueId();
    int face_flags = face.itemBase().flags();
    if (face.nbCell()==2 && (face.cell(0).level()==0 && face.cell(1).level()==0)){
      if ( (face.cell(0).owner()!=sid || face.cell(1).owner()!= sid) ||
           (face_flags & ItemFlags::II_Shared) ||
           (face_flags & ItemFlags::II_SubDomainBoundary)){
        m_shared_face_uids.add(face_uid);
      }
      else if (_hasSharedNodes(face)){
        m_connected_shared_face_uids.add(face_uid);
      }
    }
    else if (face.nbCell()==1 && face.cell(0).level() == 0){
      if (_hasSharedNodes(face)){
        m_connected_shared_face_uids.add(face_uid);
      }
    }
  }
  m_is_updated = true ;

  CHECKPERF( m_perf_counter.stop(PerfCounter::INIT) )
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelAMRConsistency::
_addFaceToList(Face face, FaceInfoMap& face_map)
{
  Integer nb_node = face.nbNode();
  Real3 center(0., 0., 0.);
  Integer data_index = m_face_info_mng.size();
  for (Node node : face.nodes()){
    Real3 node_coord = m_nodes_coord[node];
    ItemUniqueId uid = node.uniqueId();
    m_face_info_mng.add(uid);
    center += node_coord;
  }
  center /= nb_node;
  //info() << "ADD FACE uid=" << face.uniqueId() << " nb_node="
  //<< nb_node << " center=" << center;
  FaceInfo fi(face.uniqueId(), face.cell(0).uniqueId(), nb_node, face.owner(), data_index, &m_face_info_mng);
  fi.setCenter(center);
  face_map.add(face.uniqueId(), fi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParallelAMRConsistency::
_hasSharedNodes(Face face)
{
  //CHECK that one edge is connected to a ghost cell
  for ( Node node : face.nodes() ){
    if (node.itemBase().flags() &  ItemFlags::II_Shared)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine les faces à envoyer aux voisins.
 *
 * Envoie à tous les sous-domaine les faces de numéros uniques
 * et réceptionne celles de tous les autres sous-domaines.
 */
void ParallelAMRConsistency::
makeNewItemsConsistent(NodeMapCoordToUid& node_finder, FaceMapCoordToUid& face_finder)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::COMPUTE) )
  //Integer nb_sub_domain_boundary_face = 0;
  // Marque les noeuds sur la frontière
  Integer nb_active_face = static_cast<Integer> (m_mesh->nbFace() * 0.2); // 20% de faces shared (sur estimé)

  m_nodes_info.resize((nb_active_face * 2) + 5);
  m_nodes_info.clear() ;
  ENUMERATE_NODE(i_item,m_mesh->allNodes()){
    Real3 node_coord = m_nodes_coord[i_item];
    ItemUniqueId uid = i_item->uniqueId();
    NodeInfo node_info(uid, i_item->owner());
    node_info.setCoord(node_coord);
    m_nodes_info.add(uid, node_info);
  }


  m_active_faces.resize((nb_active_face * 2) + 5);
  m_active_nodes.resize((nb_active_face * 2) + 5);
  m_active_faces.clear() ;
  m_face_info_mng.m_nodes_unique_id.clear() ;
  m_face_info_mng.m_nodes_unique_id.reserve(nb_active_face) ;


  FaceFamily* true_face_family = dynamic_cast<FaceFamily*> (m_mesh->faceFamily());
  if (!true_face_family)
    ARCANE_FATAL("can not obtain FaceFamily");

  Int32 sid = m_mesh->parallelMng()->commRank();
  //UniqueArray<ItemInternal*> active_faces;
  ItemMap active_faces ;
  UniqueArray < ItemUniqueId > active_faces_to_send;
  typedef std::set<ItemInternal*> Set;
  Set active_nodes_set;
  ItemMap active_nodes ;
  // Parcours les faces et marque les noeuds frontieres actives
  DynamicMesh* mesh = dynamic_cast<DynamicMesh*> (m_mesh);
  if (!mesh)
    throw FatalErrorException(A_FUNCINFO, "can not obtain DynamicMesh");
  ItemInternalMap& faces_map = mesh->facesMap();

  for(Integer iface=0;iface<m_shared_face_uids.size();++iface){
    Int64 face_uid = m_shared_face_uids[iface];
    Face face = faces_map.findItem(face_uid);
    {
      UniqueArray<ItemInternal*> subfaces;
      true_face_family->allSubFaces(face, subfaces);
      if (subfaces.size() != 1){
        for (Integer s = 0; s < subfaces.size(); s++){
          Face face2 = subfaces[s];
          face2.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
          Int64 uid = face2.uniqueId() ;
          bool face_to_send = face_finder.isNewUid(uid);
          if(face_to_send){
            _addFaceToList(face2, m_active_faces);

            //active_faces.add(face2);
            active_faces.insert(ItemMapValue(uid,face2));
            active_faces_to_send.add(ItemUniqueId(uid));
            for ( Node node : face2.nodes() ){
              node.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
              active_nodes.insert(ItemMapValue(node.uniqueId(),node));
            }
            for ( Edge edge : face2.edges() ){
              edge.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
            }
          }
        }
      }
    }
  }
  for(Integer iface=0;iface<m_connected_shared_face_uids.size();++iface){
    Int64 face_uid = m_connected_shared_face_uids[iface];
    Face face(faces_map.findItem(face_uid));

    {
      Integer nb_node = face.nbNode() ;
      typedef std::pair<Real3,Real3> Edge ;
      UniqueArray<Edge> edges ;
      edges.reserve(nb_node) ;
      for(Integer i=0;i<nb_node;++i){
        Integer next = i==nb_node-1?0:i+1;
        Node node1 = face.node(i) ;
        Node node2 = face.node(next) ;
        if( (node1.itemBase().flags() & (ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary)) &&
            (node2.itemBase().flags() & (ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary) ) ){
          edges.add(Edge(m_nodes_coord[node1],m_nodes_coord[node2]-m_nodes_coord[node1])) ;
        }
      }
      UniqueArray<ItemInternal*> subfaces;
      true_face_family->allSubFaces(face, subfaces);
      for (Integer s = 0; s < subfaces.size(); s++){
        Face face2(subfaces[s]);
        Int64 uid = face2.uniqueId() ;
        bool face_to_send = face_finder.isNewUid(uid) ;
        if (face_to_send){
          Integer nb_node2 = face2.nbNode() ;
          for(Integer i=0;i<nb_node2;++i){
            Node node_i = face2.node(i) ;
            Real3 Xi = m_nodes_coord[node_i] ;
            for(Integer j=0;j<edges.size();++j){
              Real3 n = Xi-edges[j].first ;
              Real sinteta = math::cross(edges[j].second,n).squareNormL2() ;
              if (math::isZero(sinteta)){
                node_i.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
                //active_nodes_set.insert(node_i);
                active_nodes.insert(ItemMapValue(node_i.uniqueId(),node_i));
                _addNodeToList(node_i, m_active_nodes);
              }
            }
          }
        }
      }
    }
  }

  UniqueArray<ItemUniqueId> active_nodes_to_send(arcaneCheckArraySize(active_nodes.size()));
  ItemMap::const_iterator nit(active_nodes.begin()), nend(active_nodes.end());
  for (int i=0; nit != nend; ++nit,++i){
    Item node = nit->second;
    active_nodes_to_send[i]= node.uniqueId();
    //active_nodes.insert(ItemMapValue(uid,node)) ;
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::COMPUTE) )

  //CHECKPERF( m_perf_counter.start(PerfCounter::GATHERFACE) )
  ItemUidSet update_face_uids ;
  ItemUidSet update_node_uids ;
  _gatherFaces(active_faces_to_send,active_nodes_to_send, m_active_faces, node_finder, face_finder,update_face_uids,update_node_uids);
  //CHECKPERF( m_perf_counter.stop(PerfCounter::GATHERFACE) )



  CHECKPERF( m_perf_counter.start(PerfCounter::UPDATE) )
  //UPDATE FACES
  //for (Integer index = 0; index < active_faces.size(); index++)
  for(ItemUidSet::iterator iter = update_face_uids.begin();iter!=update_face_uids.end();++iter){
    Int64 face_uid = *iter ;
    Item face = active_faces[face_uid];
    //const Int64 current_uid = face->uniqueId();
    FaceInfo& fi = m_active_faces[face.uniqueId()];
    Int64 new_uid = fi.uniqueId() ;
    faces_map.remove(face_uid) ;
    //if (current_uid != fi.uniqueId())
    ARCANE_ASSERT((face_uid != new_uid),("AMR CONSISTENCY UPDATE FACE ERROR")) ;
    face.mutableItemBase().setUniqueId(fi.uniqueId());
    face.mutableItemBase().setOwner(fi.owner(), sid);
    //debug() << "[\t ParallelAMRConsistency] NEW FACE BEFORE " << face->uniqueId()<<" new uid "<<fi.uniqueId() << " " << fi.owner();
    faces_map.add(new_uid,ItemCompatibility::_itemInternal(face));
  }

  //UPDATE NODES
  //ENUMERATE_NODE(i_item,m_mesh->allNodes())
  ItemInternalMap& nodes_map = mesh->nodesMap();
  for(ItemUidSet::iterator iter = update_node_uids.begin();iter!=update_node_uids.end();++iter){
    //ItemInternal* node = i_item->internal() ;
    Int64 node_uid = *iter ;
    Item node = active_nodes[node_uid];
    if (node.null())
      ARCANE_FATAL("AMR CONSISTENCY NULL NODE ERROR");
    //const Int64 node_uid = node->uniqueId();
    NodeInfo ni = m_nodes_info[node.uniqueId()];
    Int64 new_uid = ni.uniqueId() ;
    //if (node_uid != ni.uniqueId())
    ARCANE_ASSERT((node_uid != new_uid),("AMR CONSISTENCY UPDATE NODE ERROR")) ;
    //debug() << "[\t ParallelAMRConsistency] OLD NEW NODE " << node_uid << " " << ni.uniqueId()<<" owner "<<ni.owner();
    nodes_map.remove(node_uid) ;
    node.mutableItemBase().setUniqueId(ni.uniqueId());
    node.mutableItemBase().setOwner(ni.owner(), sid);
    node.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
    nodes_map.add(new_uid,ItemCompatibility::_itemInternal(node));
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::UPDATE) )

  CHECKPERF( m_perf_counter.start(PerfCounter::ENDUPDATE) )
  m_mesh->nodeFamily()->partialEndUpdate();
  m_mesh->faceFamily()->partialEndUpdate();
  CHECKPERF( m_perf_counter.stop(PerfCounter::ENDUPDATE) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning Cette méthode ne doit pas être appelée en séquentiel.
 *
 * Envoie à tous les sous-domaine les faces de numéros uniques
 * \a faces_to_send de la liste \a face_map et réceptionne
 * celles de tous les autres sous-domaines.
 */
void ParallelAMRConsistency::
_gatherFaces(ConstArrayView<ItemUniqueId> faces_to_send,
             ConstArrayView<ItemUniqueId> nodes_to_send,
             FaceInfoMap& face_map,
             MapCoordToUid& node_finder,
             MapCoordToUid& face_finder,
             ItemUidSet& updated_face_uids,
             ItemUidSet& updated_node_uids)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::GATHERFACE) )

  IParallelMng* pm = m_mesh->parallelMng();
  Integer sub_domain_id = pm->commRank();
  Integer nb_sub_domain = pm->commSize();

  const Real tol = 10e-12;

  SerializeBuffer sbuf;
  sbuf.setMode(ISerializer::ModeReserve);
  Integer nb_to_send = faces_to_send.size();
  Integer nb_node_to_send = nodes_to_send.size();

  Int64UniqueArray unique_ids(nb_to_send);
  Int64UniqueArray cells_unique_ids(nb_to_send);
  Int64UniqueArray nodes_unique_id;
  nodes_unique_id.reserve(nb_node_to_send);
  RealUniqueArray coords;
  coords.reserve(3 * nb_to_send);
  RealUniqueArray nodes_coords;
  nodes_coords.reserve(3 *nb_node_to_send );
  for (Integer i = 0; i < nb_to_send; ++i){
    const FaceInfo& fi = face_map[faces_to_send[i]];
    unique_ids[i] = fi.uniqueId().asInt64();
    cells_unique_ids[i] = fi.cellUniqueId().asInt64();
    coords.add(fi.center().x);
    coords.add(fi.center().y);
    coords.add(fi.center().z);
  }
  debug()<<"SEND NODE : "<<nb_node_to_send;
  for (Integer i = 0; i < nb_node_to_send; ++i){
      //debug()<<"SEND NODE : "<<nodes_to_send[i];
      ItemUniqueId nuid(nodes_to_send[i]);
      nodes_unique_id.add(nuid.asInt64());
      NodeInfo ni = m_nodes_info[nuid];
      Real3 c = ni.getCoord();
      nodes_coords.add(c.x);
      nodes_coords.add(c.y);
      nodes_coords.add(c.z);
  }
  sbuf.reserveInteger(1); // pour le nombre de faces
  sbuf.reserveInteger(1); // pour le numéro du sous-domaine
  sbuf.reserveInteger(1); // pour le nombre de noeuds dans la liste
  sbuf.reserveArray(unique_ids); // pour le unique id des faces
  sbuf.reserveArray(cells_unique_ids); // pour le unique id des mailles des faces
  sbuf.reserveArray(nodes_unique_id); // pour la liste des noeuds
  sbuf.reserveArray(coords); // pour les coordonnées du centre
  sbuf.reserveArray(nodes_coords); // pour les coordonnées des noeuds

  sbuf.allocateBuffer();
  sbuf.setMode(ISerializer::ModePut);

  sbuf.putInteger(nb_to_send);
  sbuf.putInteger(sub_domain_id);
  sbuf.putInteger(nodes_unique_id.size());
  sbuf.putArray(unique_ids);
  sbuf.putArray(cells_unique_ids);
  sbuf.putArray(nodes_unique_id);
  sbuf.putArray(coords);
  sbuf.putArray(nodes_coords);

  SerializeBuffer recv_buf;
  pm->allGather(&sbuf, &recv_buf);
  recv_buf.setMode(ISerializer::ModeGet);

  for (Integer i = 0; i < nb_sub_domain; ++i){
    Integer nb_face = recv_buf.getInteger();
    Integer sid = recv_buf.getInteger();
    Integer nb_node_unique_id = recv_buf.getInteger();
    info() << " [\t ParallelAMRConsistency::_gatherFaces] READ nface=" << nb_face << " FROM sid=" << sid<<" "<<m_face_info_mng.size();
    recv_buf.getArray(unique_ids);
    recv_buf.getArray(cells_unique_ids);
    recv_buf.getArray(nodes_unique_id);
    recv_buf.getArray(coords);
    recv_buf.getArray(nodes_coords);

    // Parcours toutes les faces reçues si certaines sont absentes,
    // on les ignore.
    for (Integer z = 0; z < nb_face; ++z){
      ItemUniqueId new_uid(unique_ids[z]);
      ItemUniqueId cell_uid(cells_unique_ids[z]);

      Real3 center;
      center.x = coords[z * 3];
      center.y = coords[z * 3 + 1];
      center.z = coords[z * 3 + 2];

      const Int64 current_uid = face_finder.find(center, tol);
      if ((current_uid != NULL_ITEM_ID) && (new_uid < current_uid)){
        if (!face_map.hasKey(ItemUniqueId(current_uid))){
          error() << "face uid not found \n";
        }
        //UPDATE FACE_FINDER
        face_finder.insert(center,new_uid,tol) ;
        updated_face_uids.insert(current_uid) ;

        FaceInfo& fi_old = face_map[ItemUniqueId(current_uid)];
        Integer data_index = fi_old.getDataIndex();
        Integer nb_node = fi_old.nbNode() ;
        FaceInfo fi(new_uid, cell_uid, nb_node, sid, data_index, &m_face_info_mng);
        fi.setCenter(center);
        face_map[ItemUniqueId(current_uid)] = fi;
      }
    }

    for (Integer z = 0; z < nb_node_unique_id; ++z){
      ItemUniqueId nuid(nodes_unique_id[z]);
      Real3 node_coord;
      node_coord.x = nodes_coords[z * 3];
      node_coord.y = nodes_coords[z * 3 + 1];
      node_coord.z = nodes_coords[z * 3 + 2];
      Int64 current_node_uid = node_finder.find(node_coord, tol);
      if ((current_node_uid != NULL_ITEM_ID) && (nuid < current_node_uid)){
        ItemUniqueId c_nuid(current_node_uid);
        if(nuid<m_nodes_info[c_nuid].uniqueId()){
          NodeInfo ni(nuid, sid);
          ni.setCoord(node_coord);
          m_nodes_info[c_nuid] = ni;

          // UPDATE NODE FINDER
          updated_node_uids.insert(current_node_uid) ;
        }
      }
    }
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::GATHERFACE) )

  CHECKPERF( m_perf_counter.start(PerfCounter::UPDATE) )
  //update node_finder
  for(ItemUidSet::iterator iter = updated_node_uids.begin();iter!=updated_node_uids.end();++iter){
    ItemUniqueId uid(*iter);
    NodeInfo& ni = m_nodes_info[uid] ;
    node_finder.insert(ni.getCoord(),ni.uniqueId(),tol) ;
  }

  //update m_face_info_mng
  _update(m_face_info_mng.m_nodes_unique_id,m_nodes_info) ;

  CHECKPERF( m_perf_counter.stop(PerfCounter::UPDATE) )

  //pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelAMRConsistency::
_update(Array<ItemUniqueId>& nodes_unique_id, NodeInfoList const& nodes_info)
{
  for(Integer i=0, n=nodes_unique_id.size();i<n;++i)
  {
    ItemUniqueId& uid = nodes_unique_id[i] ;
    NodeInfoList::Data const* data = nodes_info.lookup(uid);
    if (data)
      nodes_unique_id[i] = data->value().uniqueId() ;
    else
      info()<<"ERROR "<<i<<" "<<uid<<" not found";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelAMRConsistency::
_addNodeToList(Node node, NodeInfoList& node_map)
{
  Real3 node_coord = m_nodes_coord[node];
  ItemUniqueId uid = node.uniqueId();
  NodeInfoList::Data* i = node_map.lookup(uid);
  if (!i){
    NodeInfo node_info(uid, node.owner());
    node_info.setCoord(node_coord);
    node_map.add(uid, node_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelAMRConsistency::
_addFaceToList2(Face face, FaceInfoMap2& face_map)
{
  Integer nb_node = face.nbNode();
  ItemUniqueId uid = face.uniqueId();
  FaceInfoMap2::Data* i = face_map.lookup(uid);
  if (!i){
    Real3 center(0., 0., 0.);
    for (Node node : face.nodes() ){
      Real3 node_coord = m_nodes_coord[node];
      center += node_coord;
    }
    center /= nb_node;
    FaceInfo2 fi(face.uniqueId(), face.owner());
    fi.setCenter(center);
    face_map.add(uid, fi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine les faces à envoyer aux voisins.
 *
 * Envoie à tous les sous-domaine les faces de numéros uniques
 * et réceptionne celles de tous les autres sous-domaines.
 */
void ParallelAMRConsistency::
makeNewItemsConsistent2(MapCoordToUid& node_finder, MapCoordToUid& face_finder)
{
  // Marque les noeuds sur la frontière
  Integer nb_active_face = static_cast<Integer> (m_mesh->nbFace() * 0.2); // 20% de faces shared (sur estimé)
  m_active_nodes.resize((nb_active_face * 2) + 5);
  m_active_faces.resize((nb_active_face * 2) + 5);

  FaceFamily* true_face_family = dynamic_cast<FaceFamily*> (m_mesh->itemFamily(IK_Face));
  if (!true_face_family)
    throw FatalErrorException(A_FUNCINFO, "can not obtain FaceFamily");

  Int32 sid = m_mesh->parallelMng()->commRank();


  typedef std::set<Item> Set;
  Set active_nodes_set, active_faces_set;
  // Parcours les faces et marque les noeuds frontieres actives
  DynamicMesh* mesh = dynamic_cast<DynamicMesh*> (m_mesh);
  if (!mesh)
    throw FatalErrorException(A_FUNCINFO, "can not obtain DynamicMesh");
  ItemInternalMap& faces_map = mesh->facesMap();

  faces_map.eachItem([&](Face face) {
    bool is_sub_domain_boundary_face = false;
    if (face.itemBase().flags() & ItemFlags::II_SubDomainBoundary){
      is_sub_domain_boundary_face = true; // true is not needed
    }
    else{
      if (face.cell(0).level() == 0 && face.cell(1).level() == 0){
        if ((face.cell(0).owner() != sid || face.cell(1).owner() != sid))
          is_sub_domain_boundary_face = true;
      }
    }

    if (is_sub_domain_boundary_face){
      UniqueArray<ItemInternal*> subfaces;
      true_face_family->allSubFaces(face, subfaces);
      for (Integer s = 0; s < subfaces.size(); s++){
        Face face2 = subfaces[s];
        face2.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
        _addFaceToList2(face2, m_active_faces2);
        active_faces_set.insert(face2);
        for ( Node node : face2.nodes() ){
          node.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
          active_nodes_set.insert(node);
          _addNodeToList(node, m_active_nodes);
        }
        for ( Edge edge : face2.edges() )
          edge.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
      }
    }
  });

  UniqueArray<ItemUniqueId> active_faces_to_send(arcaneCheckArraySize(active_faces_set.size()));
  UniqueArray<ItemUniqueId> active_nodes_to_send(arcaneCheckArraySize(active_nodes_set.size()));

  UniqueArray<Item> active_faces(arcaneCheckArraySize(active_faces_set.size()));
  UniqueArray<Item> active_nodes(arcaneCheckArraySize(active_nodes_set.size()));

  Set::const_iterator fit(active_faces_set.begin()), fend(active_faces_set.end());
  Integer i=0;
  for (; fit != fend; ++fit){
    Item face = *fit;
    active_faces_to_send[i]=(face.uniqueId());
    active_faces[i]=face;
    i++;
  }
  Set::const_iterator nit(active_nodes_set.begin()), nend(active_nodes_set.end());
  i=0;
  for (; nit != nend; ++nit){
    Item node = *nit;
    active_nodes_to_send[i]= (node.uniqueId());
    //debug() << "ACTIVE NODE TO SEND " << node->uniqueId() << " " << active_nodes_to_send[i];
    active_nodes[i]=node;
    i++;
  }

  _gatherItems(active_nodes_to_send, active_faces_to_send, m_active_nodes, m_active_faces2, node_finder, face_finder);

  for (Integer index = 0; index < active_faces.size(); index++){
    Item face = active_faces[index];
    const Int64 current_uid = face.uniqueId();
    FaceInfo2& fi = m_active_faces2[face.uniqueId()];
    if (current_uid != fi.uniqueId()){
      face.mutableItemBase().setUniqueId(fi.uniqueId());
      face.mutableItemBase().setOwner(fi.owner(), sid);
      //debug() << "[\t ParallelAMRConsistency] NEW FACE BEFORE " << fi.uniqueId() << " " << fi.owner();
    }
  }
  for (Integer index = 0; index < active_nodes.size(); index++){
    Item node = active_nodes[index];
    const Int64 current_uid = node.uniqueId();
    NodeInfo& ni = m_active_nodes[node.uniqueId()];
    if (current_uid != ni.uniqueId()){
      node.mutableItemBase().setUniqueId(ni.uniqueId());
      node.mutableItemBase().setOwner(ni.owner(), sid);
      //debug() << "[\t ParallelAMRConsistency] NEW FACE BEFORE " << fi.uniqueId() << " " << fi.owner();
    }
  }

  // Il faut ranger à nouveau #m_faces_map car les uniqueId() des
  // faces ont été modifiés
  faces_map.notifyUniqueIdsChanged();
  // idem pour les noeuds
  mesh->nodesMap().notifyUniqueIdsChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning Cette méthode ne doit pas être appelée en séquentiel.
 *
 * Envoie à tous les sous-domaine les faces de numéros uniques
 * \a faces_to_send de la liste \a face_map et réceptionne
 * celles de tous les autres sous-domaines.
 */
void ParallelAMRConsistency::
_gatherItems(ConstArrayView<ItemUniqueId> nodes_to_send,
             ConstArrayView<ItemUniqueId> faces_to_send,
             NodeInfoList& node_map,
             FaceInfoMap2& face_map,
             MapCoordToUid& node_finder,
             MapCoordToUid& face_finder)
{
  IParallelMng* pm = m_mesh->parallelMng();
  Integer sub_domain_id = pm->commRank();
  Integer nb_sub_domain = pm->commSize();

  SerializeBuffer sbuf;
  sbuf.setMode(ISerializer::ModeReserve);
  Integer nb_to_send = faces_to_send.size();
  Integer nb_node_to_send = nodes_to_send.size();
  Int64UniqueArray unique_ids(nb_to_send);
  Int64UniqueArray node_unique_ids(nb_node_to_send);
  RealUniqueArray coords;
  coords.reserve(3 * nb_to_send);
  RealUniqueArray nodes_coords;
  nodes_coords.reserve(3 * nb_node_to_send);

  for (Integer i = 0; i < nb_node_to_send; ++i){
    const NodeInfo& ni = node_map[nodes_to_send[i]];
    node_unique_ids[i] = ni.uniqueId().asInt64();
    const Real3 c = ni.getCoord();
    nodes_coords.add(c.x);
    nodes_coords.add(c.y);
    nodes_coords.add(c.z);
  }

  for (Integer i = 0; i < nb_to_send; ++i){
    const FaceInfo2& fi = face_map[faces_to_send[i]];
    unique_ids[i] = fi.uniqueId().asInt64();
    coords.add(fi.center().x);
    coords.add(fi.center().y);
    coords.add(fi.center().z);
  }

  sbuf.reserveInteger(1); // pour le nombre de faces
  sbuf.reserveInteger(1); // pour le numéro du sous-domaine
  sbuf.reserveInteger(1); // pour le nombre de noeuds dans la liste
  sbuf.reserveArray(unique_ids); // pour le unique id des faces
  sbuf.reserveArray(node_unique_ids); // pour la liste des noeuds
  sbuf.reserveArray(coords); // pour les coordonnées du centre
  sbuf.reserveArray(nodes_coords); // pour les coordonnées des noeuds

  sbuf.allocateBuffer();
  sbuf.setMode(ISerializer::ModePut);

  sbuf.putInteger(nb_to_send);
  sbuf.putInteger(sub_domain_id);
  sbuf.putInteger(node_unique_ids.size());
  sbuf.putArray(unique_ids);
  sbuf.putArray(node_unique_ids);
  sbuf.putArray(coords);
  sbuf.putArray(nodes_coords);

  SerializeBuffer recv_buf;
  pm->allGather(&sbuf, &recv_buf);
  recv_buf.setMode(ISerializer::ModeGet);

  for (Integer i = 0; i < nb_sub_domain; ++i){
    Integer nb_face = recv_buf.getInteger();
    Integer sid = recv_buf.getInteger();
    Integer nb_node_unique_id = recv_buf.getInteger();
    //info() << " [\t ParallelAMRConsistency::_gatherFaces] READ nface=" << nb_face << " FROM sid=" << sid;

    recv_buf.getArray(unique_ids);
    recv_buf.getArray(node_unique_ids);
    recv_buf.getArray(coords);
    recv_buf.getArray(nodes_coords);

    // Parcours toutes les faces reçues si certaines sont absentes,
    // on les ignore.
    const Real tol = 10e-6;
    for (Integer z = 0; z < nb_face; ++z){
      ItemUniqueId new_uid(unique_ids[z]);

      Real3 center;
      center.x = coords[z * 3];
      center.y = coords[z * 3 + 1];
      center.z = coords[z * 3 + 2];

      const Int64 current_uid = face_finder.find(center, tol);
      if ((current_uid != NULL_ITEM_ID) && (new_uid < current_uid)){
        if (!face_map.hasKey(ItemUniqueId(current_uid))){
          error() << "face uid not found \n";
        }
        FaceInfo2 fi(new_uid, sid);
        fi.setCenter(center);
        face_map[ItemUniqueId(current_uid)] = fi;
      }
    }

    for (Integer z = 0; z < nb_node_unique_id; ++z){
      ItemUniqueId nuid(node_unique_ids[z]);
      Real3 node_coord;
      node_coord.x = nodes_coords[z * 3];
      node_coord.y = nodes_coords[z * 3 + 1];
      node_coord.z = nodes_coords[z * 3 + 2];
      Int64 current_node_uid = node_finder.find(node_coord, tol);
      if ((current_node_uid != NULL_ITEM_ID) && (nuid < current_node_uid)){
        NodeInfo ni(nuid, sid);
        ni.setCoord(node_coord);
        ItemUniqueId c_nuid(current_node_uid);
        node_map[c_nuid] = ni;
      }
    }
  }
  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
