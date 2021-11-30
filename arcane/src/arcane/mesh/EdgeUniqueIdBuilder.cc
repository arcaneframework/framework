﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeUniqueIdBuilder.cc                                      (C) 2000-2021 */
/*                                                                           */
/* Construction des indentifiants uniques des edges.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/EdgeUniqueIdBuilder.h"
#include "arcane/mesh/GhostLayerBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"

#include "arcane/IParallelExchanger.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/ISerializer.h"
#include "arcane/ParallelMngUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeUniqueIdBuilder::
EdgeUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeUniqueIdBuilder::
~EdgeUniqueIdBuilder()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeUniqueIdBuilder::
computeEdgesUniqueIds()
{
  //Integer sid = pm->commRank();
  Int64 begin_time = platform::getCPUTime();

    _computeEdgesUniqueIdsParallel3();

//   if (pm->isParallel()){
//     _computeEdgesUniqueIdsParallel3();
//   }
//   else{
//     if (!platform::getEnvironmentVariable("ARCANE_NO_EDGE_RENUMBER").null()){
//       pwarning() << "No edge renumbering";
//       return;
//     }
//     _computeEdgesUniqueIdsSequential();
//   }

  Int64 end_time = platform::getCPUTime();
  Real diff = (Real)(end_time - begin_time);
  info() << "TIME to compute edge unique ids=" << (diff/1.0e6);

  ItemInternalMap& edges_map = m_mesh->edgesMap();

  // Il faut ranger à nouveau #m_edges_map car les uniqueId() des
  // edges ont été modifiés
  edges_map.notifyUniqueIdsChanged();

  if (m_mesh_builder->isVerbose()){
    info() << "NEW EDGES_MAP after re-indexing";
    ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,edges_map){
      ItemInternal* edge = nbid->value();
      info() << "Edge uid=" << edge->uniqueId() << " lid=" << edge->localId();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'aide pour la détermination en parallèle
 * des unique_id des edges.
 *
 * \note Tous les champs de cette classe doivent être de type Int64
 * car elle est sérialisée par cast en Int64*.
 */
class T_CellEdgeInfo
{
 public:

  T_CellEdgeInfo(Int64 uid,Integer nb_back_edge,Integer nb_true_boundary_edge)
  : m_unique_id(uid), m_nb_back_edge(nb_back_edge), m_nb_true_boundary_edge(nb_true_boundary_edge)
  {
  }

  T_CellEdgeInfo()
  : m_unique_id(NULL_ITEM_ID), m_nb_back_edge(0), m_nb_true_boundary_edge(0)
  {
  }

 public:

  bool operator<(const T_CellEdgeInfo& ci) const
    {
      return m_unique_id<ci.m_unique_id;
    }

 public:

  Int64 m_unique_id;
  Int64 m_nb_back_edge;
  Int64 m_nb_true_boundary_edge;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * //COPIE DEPUIS GhostLayerBuilder.
 * Faire une classe unique.
 */
void EdgeUniqueIdBuilder::
_exchangeData(IParallelExchanger* exchanger,BoundaryInfosMap& boundary_infos_to_send)
{
  for( BoundaryInfosMapEnumerator i_map(boundary_infos_to_send); ++i_map; ){
    Int32 sd = i_map.data()->key();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  {
    for( Integer i=0, ns=exchanger->nbSender(); i<ns; ++i ){
      ISerializeMessage* sm = exchanger->messageToSend(i);
      Int32 rank = sm->destination().value();
      ISerializer* s = sm->serializer();
      Int64ConstArrayView infos  = boundary_infos_to_send[rank];
      Integer nb_info = infos.size();
      s->setMode(ISerializer::ModeReserve);
      s->reserve(DT_Int64,1); // Pour le nombre d'elements
      s->reserveSpan(DT_Int64,nb_info); // Pour les elements
      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);
      //info() << " SEND1 rank=" << rank << " nb_info=" << nb_info;
      s->putInt64(nb_info);
      s->putSpan(infos);
    }
  }
  exchanger->processExchange();
  debug() << "END EXCHANGE";
}

//class EdgeNodeList
//{
//Int64 nodes[ItemSharedInfo::MAX_EDGE_NODE];
//};

template<typename DataType>
class ItemInfoMultiList
{
 public:
 private:

  class MyInfo
  {
   public:
    MyInfo(const DataType& d,Integer n) : data(d), next_index(n) {}
   public:
    DataType data;
    Integer next_index;
  };

 public:
  ItemInfoMultiList() : m_last_index(5000,true) {}

 public:

  void add(Int64 node_uid,const DataType& data)
  {
    Integer current_index = m_values.size();

    bool is_add = false;
    HashTableMapT<Int64,Int32>::Data* d = m_last_index.lookupAdd(node_uid,-1,is_add);

    m_values.add(MyInfo(data,d->value()));
    d->value() = current_index;
  }

 public:
  UniqueArray<MyInfo> m_values;
  HashTableMapT<Int64,Int32> m_last_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque edge en parallèle.
  
  NEW VERSION.
*/  
void EdgeUniqueIdBuilder::
_computeEdgesUniqueIdsParallel3()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Integer my_rank = pm->commRank();
  Integer nb_rank = pm->commSize();

  Integer nb_local_edge = m_mesh_builder->oneMeshItemAdder()->nbEdge();
  //Integer nb_local_cell = m_mesh_builder->nbCell();
  //bool is_verbose = m_mesh_builder->isVerbose();
  
  Int64UniqueArray edges_opposite_cell_uid(nb_local_edge);
  edges_opposite_cell_uid.fill(NULL_ITEM_ID);
  IntegerUniqueArray edges_opposite_cell_index(nb_local_edge);
  IntegerUniqueArray edges_opposite_cell_owner(nb_local_edge);

  // Pour vérification, s'assure que tous les éléments de ce tableau
  // sont valides, ce qui signifie que toutes les edges ont bien été
  // renumérotés
  Int64UniqueArray edges_new_uid(nb_local_edge);
  edges_new_uid.fill(NULL_ITEM_ID);

  //UniqueArray<T_EdgeInfo> edges_local_infos(m_mesh_nb_edge);

  //Integer nb_recv_sub_domain_boundary_edge = 0;

  Int64UniqueArray edges_infos;
  edges_infos.reserve(10000);
  //ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& edges_map = m_mesh->edgesMap();
  ItemInternalMap& faces_map = m_mesh->facesMap(); // utilisé pour détecter le bord
  ItemInternalMap& nodes_map = m_mesh->nodesMap();


  // NOTE: ce tableau n'est pas utile sur toutes les mailles. Il
  // suffit qu'il contienne les mailles dont on a besoin, c'est à dire
  // les notres + celles connectées à une de nos edges.
  HashTableMapT<Int32,Int32> cell_first_edge_uid(m_mesh_builder->oneMeshItemAdder()->nbCell()*2,true);
  //OLD Int64UniqueArray cell_first_edge_uid(global_max_cell_uid+1);
  //OLD cell_first_edge_uid.fill(0);

  // Rassemble les données des autres processeurs dans recv_cells;
  // Pour éviter que les tableaux ne soient trop gros, on procède en plusieurs
  // étapes.
  // Chaque sous-domaine construit sa liste de edges frontières, avec pour
  // chaque edge:
  // - son type
  // - la liste de ses noeuds,
  // - le numéro unique de sa maille
  // - le propriétaire de sa maille
  // - son indice dans sa maille
  // Cette liste sera ensuite envoyée à tous les sous-domaines.
  ItemTypeMng* itm = m_mesh->itemTypeMng();

  // Détermine le unique id max des noeuds
  Int64 my_max_node_uid = NULL_ITEM_UNIQUE_ID;
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,nodes_map){
    ItemInternal* node = nbid->value();
    Int64 node_uid = node->uniqueId();
    if (node_uid>my_max_node_uid)
      my_max_node_uid = node_uid;
  }
  Int64 global_max_node_uid = pm->reduce(Parallel::ReduceMax,my_max_node_uid);
  debug() << "NODE_UID_INFO: MY_MAX_UID=" << my_max_node_uid
         << " GLOBAL=" << global_max_node_uid;
 
  //TODO: choisir bonne valeur pour initialiser la table
  BoundaryInfosMap boundary_infos_to_send(nb_rank,true);
  NodeUidToSubDomain uid_to_subdomain_converter(global_max_node_uid,nb_rank);

  HashTableMapT<Int64,SharedArray<Int64> > nodes_info(100000,true);
  IItemFamily* node_family = m_mesh->nodeFamily();
  UniqueArray<bool> is_boundary_nodes(node_family->maxLocalId(),false);

  // Marque tous les noeuds frontieres car ce sont ceux qu'il faudra envoyer
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,faces_map){
    ItemInternal* face = nbid->value();
    Integer face_nb_cell = face->nbCell();
    if (face_nb_cell==1){
      for( Int32 ilid : face->internalNodes().localIds() )
        is_boundary_nodes[ilid] = true;
    }
  }

  // Détermine la liste des edges frontières
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,edges_map){
    Edge edge = nbid->value();
    Node first_node = edge.node(0);
    Int64 first_node_uid = first_node.uniqueId();
    SharedArray<Int64> v;
    Int32 dest_rank = -1;
    if (!is_boundary_nodes[first_node.localId()]){
      v = nodes_info.lookupAdd(first_node_uid)->value();
    }
    else{
      dest_rank = uid_to_subdomain_converter.uidToRank(first_node_uid);
      v = boundary_infos_to_send.lookupAdd(dest_rank)->value();
    }
    v.add(first_node_uid);      // 0
    v.add(my_rank);             // 1 
    v.add(edge.uniqueId());    // 2
    v.add(edge.type());      // 3
    v.add(NULL_ITEM_UNIQUE_ID); // 4 : only used for debug
    v.add(NULL_ITEM_UNIQUE_ID); // 5 : only used for debug
    for( Node edge_node : edge.nodes() )
      v.add(edge_node.uniqueId());
  }

  // Positionne la liste des envoies
  Ref<IParallelExchanger> exchanger{ParallelMngUtils::createExchangerRef(pm)};
  _exchangeData(exchanger.get(),boundary_infos_to_send);

  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    SharedArray<Int64> received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb_info = s->getInt64();
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      received_infos.resize(nb_info);
      s->getSpan(received_infos);
      //if ((nb_info % 3)!=0)
      //fatal() << "info size can not be divided by 3";
      Integer z =0; 
      while(z<nb_info){
        Int64 node_uid = received_infos[z+0];
        //Int64 sender_rank = received_infos[z+1];
        //Int64 edge_uid = received_infos[z+2];
        Int32 edge_type = (Int32)received_infos[z+3];
        // received_infos[z+4];
        // received_infos[z+5];
        ItemTypeInfo* itt = itm->typeFromId(edge_type);
        Integer edge_nb_node = itt->nbLocalNode();
        Int64Array& a = nodes_info.lookupAdd(node_uid)->value();
        a.addRange(Int64ConstArrayView(6+edge_nb_node,&received_infos[z]));
        z += 6;
        z += edge_nb_node;
        /*info() << "NODE UID=" << node_uid << " sender=" << sender_rank
          << " edge_uid=" << edge_uid */
        //node_cell_list.add(node_uid,cell_uid,cell_owner);
        //HashTableMapT<Int64,Int32>::Data* v = nodes_nb_cell.lookupAdd(node_uid);
        //++v->value();
      }
    }
    Integer my_max_edge_node = 0;
    for( HashTableMapT<Int64,SharedArray<Int64> >::Enumerator inode(nodes_info); ++inode; ){
      //Int64 key = inode.data()->key();
      Int64ConstArrayView a = *inode;
      //info() << "A key=" << key << " size=" << a.size();
      Integer nb_info = a.size();
      Integer z = 0;
      Integer node_nb_edge = 0;
      while(z<nb_info){
        ++node_nb_edge;
        //Int64 node_uid = a[z+0];
        //Int64 sender_rank = a[z+1];
        //Int64 edge_uid = a[z+2];
        Int32 edge_type = (Int32)a[z+3];
        // a[z+4];
        // a[z+5];
        ItemTypeInfo* itt = itm->typeFromId(edge_type);
        Integer edge_nb_node = itt->nbLocalNode();
        /*info() << "NODE2 UID=" << node_uid << " sender=" << sender_rank
          << " edge_uid=" << edge_uid */
        //for( Integer y=0; y<edge_nb_node; ++y )
        //info() << "Nodes = i="<< y << " " << a[z+6+y];
        z += 6;
        z += edge_nb_node;
      }
      my_max_edge_node = math::max(node_nb_edge,my_max_edge_node);
    }
    Integer global_max_edge_node = pm->reduce(Parallel::ReduceMax,my_max_edge_node);
    debug() << "GLOBAL MAX EDGE NODE=" << global_max_edge_node;
    // OK, maintenant donne comme uid de la edge (node_uid * global_max_edge_node + index)
    IntegerUniqueArray indexes;
    boundary_infos_to_send = BoundaryInfosMap(nb_rank,true);

    for( HashTableMapT<Int64,SharedArray<Int64> >::Enumerator inode(nodes_info); ++inode; ){
      //Int64 key = inode.data()->key();
      Int64ConstArrayView a = *inode;
      //info() << "A key=" << key << " size=" << a.size();
      Integer nb_info = a.size();
      Integer z = 0;
      Integer node_nb_edge = 0;
      indexes.clear();
      while(z<nb_info){
        Int64 node_uid = a[z+0];
        Int32 sender_rank = (Int32)a[z+1];
        Int64 edge_uid = a[z+2];
        Int32 edge_type = (Int32)a[z+3];
        // a[z+4];
        // a[z+5];
        ItemTypeInfo* itt = itm->typeFromId(edge_type);
        Integer edge_nb_node = itt->nbLocalNode();

        // Regarde si la edge est déjà dans la liste:
        Integer edge_index = node_nb_edge;
        Int32 edge_new_owner = sender_rank;
        for( Integer y=0; y<node_nb_edge; ++y ){
          if (memcmp(&a[indexes[y]+6],&a[z+6],sizeof(Int64)*edge_nb_node)==0){
            edge_index = y;
            edge_new_owner = (Int32)a[indexes[y]+1];
            //info() << "SAME EDGE AS y=" << y << " owner=" << edge_new_owner;
          }
        }
        Int64 edge_new_uid = (node_uid * global_max_edge_node) + edge_index;
        Int64Array& v = boundary_infos_to_send.lookupAdd(sender_rank)->value();
        // Indique au propriétaire de cette edge son nouvel uid
        v.add(edge_uid);
        v.add(edge_new_uid);
        v.add(edge_new_owner);
        indexes.add(z);
        z += 6;
        z += edge_nb_node;
        /*info() << "NODE3 UID=" << node_uid << " sender=" << sender_rank
          << " edge_uid=" << edge_uid
          << " edge_index=" << edge_index;*/
        ++node_nb_edge;
      }
      my_max_edge_node = math::max(node_nb_edge,my_max_edge_node);
    }
  }
  exchanger = ParallelMngUtils::createExchangerRef(pm);

  _exchangeData(exchanger.get(),boundary_infos_to_send);
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb_info = s->getInt64();
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      received_infos.resize(nb_info);
      s->getSpan(received_infos);
      if ((nb_info % 3)!=0)
        ARCANE_FATAL("info size can not be divided by 3 x={0}",nb_info);
      Int64 nb_item = nb_info / 3;
      for (Int64 z=0; z<nb_item; ++z ){
        Int64 old_uid = received_infos[(z*3)];
        Int64 new_uid = received_infos[(z*3)+1];
        Int32 new_owner = (Int32)received_infos[(z*3)+2];
        //info() << "EDGE old_uid=" << old_uid << " new_uid=" << new_uid;
        ItemInternalMapData* edge_data = edges_map.lookup(old_uid);
        if (!edge_data)
          fatal() << "Can not find own edge uid=" << old_uid;
        edge_data->value()->setUniqueId(new_uid);
        edge_data->value()->setOwner(new_owner,my_rank);
      }
    }
  }

  traceMng()->flush();
  pm->barrier();
  debug() << "END OF TEST NEW EDGE COMPUTE";
  return;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque edge en séquentiel.
  
  \sa computeEdgesUniqueIds()
*/  
void EdgeUniqueIdBuilder::
_computeEdgesUniqueIdsSequential()
{
  bool is_verbose = m_mesh_builder->isVerbose();
  Integer nb_cell = m_mesh_builder->oneMeshItemAdder()->nbCell();

  ItemInternalMap& cells_map = m_mesh->cellsMap();

  // En séquentiel, les uniqueId() des mailles ne peuvent dépasser la
  // taille des Integers même en 32bits.
  Int32 max_uid = 0;
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,cells_map){
    ItemInternal* cell = nbid->value();
    Int32 cell_uid = cell->uniqueId().asInt32();
    if (cell_uid>max_uid)
      max_uid = cell_uid;
  }
  info() << "Max uid=" << max_uid;
  Int32UniqueArray cell_first_edge_uid(max_uid+1);
  Int32UniqueArray cell_nb_num_back_edge(max_uid+1);
  Int32UniqueArray cell_true_boundary_edge(max_uid+1);

  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,cells_map){
    Cell cell = nbid->value();
    Int32 cell_uid = cell->uniqueId().asInt32();
    Integer nb_num_back_edge = 0;
    Integer nb_true_boundary_edge = 0;
    for( Edge edge : cell.edges()){
      if (edge.internal()->backCell()==cell)
        ++nb_num_back_edge;
      else if (edge.nbCell()==1){
        ++nb_true_boundary_edge;
      }
    }
    cell_nb_num_back_edge[cell_uid] = nb_num_back_edge;
    cell_true_boundary_edge[cell_uid] = nb_true_boundary_edge;
  }

  Integer current_edge_uid = 0;
  for( Integer i=0; i<nb_cell; ++i ){
    cell_first_edge_uid[i] = current_edge_uid;
    current_edge_uid += cell_nb_num_back_edge[i] + cell_true_boundary_edge[i];
  }
  
  if (is_verbose){
    for( Integer i=0; i<nb_cell; ++i ){
      info() << "Recv: Cell EdgeInfo celluid=" << i
             << " firstedgeuid=" << cell_first_edge_uid[i]
                    << " nbback=" << cell_nb_num_back_edge[i]
                    << " nbbound=" << cell_true_boundary_edge[i];
    }
  }

  ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,cells_map){
    Cell cell = nbid->value();
    Int32 cell_uid = cell->uniqueId().asInt32();
    Integer nb_num_back_edge = 0;
    Integer nb_true_boundary_edge = 0;
    for( Edge edge : cell.edges() ){
      Int64 edge_new_uid = NULL_ITEM_UNIQUE_ID;
      //info() << "CHECK CELLUID=" << cell_uid << " EDGELID=" << edge->localId();
      if (edge.internal()->backCell()==cell){
        edge_new_uid = cell_first_edge_uid[cell_uid] + nb_num_back_edge;
        ++nb_num_back_edge;
      }
      else if (edge.nbCell()==1){
        edge_new_uid = cell_first_edge_uid[cell_uid] + cell_nb_num_back_edge[cell_uid] + nb_true_boundary_edge;
        ++nb_true_boundary_edge;
      }
      if (edge_new_uid!=NULL_ITEM_UNIQUE_ID){
        //info() << "NEW EDGE UID: LID=" << edge->localId() << " OLDUID=" << edge->uniqueId()
        //<< " NEWUID=" << edge_new_uid << " THIS=" << edge;
        edge->internal()->setUniqueId(edge_new_uid);
      }
    }
  }

  if (is_verbose){
    OStringStream ostr;
    ENUMERATE_ITEM_INTERNAL_MAP_DATA(nbid,cells_map){
      Cell cell = nbid->value();
      Int32 cell_uid = cell.uniqueId().asInt32();
      Integer index = 0;
      for( Edge edge : cell.edges() ){
        Int64 opposite_cell_uid = NULL_ITEM_UNIQUE_ID;
        bool true_boundary = false;
        bool internal_other = false;
        if (edge->internal()->backCell()==cell){
        }
        else if (edge.nbCell()==1){
          true_boundary = true;
        }
        else{
          internal_other = true;
          opposite_cell_uid = edge.internal()->backCell()->uniqueId().asInt64();
        }
        ostr() << "NEW LOCAL ID FOR CELLEDGE " << cell_uid << ' '
               << index << ' ' << edge.uniqueId() << " (";
        for( Node node : edge.nodes() ){
          ostr() << ' ' << node.uniqueId();
        }
        ostr() << ")";
        if (internal_other)
          ostr() << " internal-other";
        if (true_boundary)
          ostr() << " true-boundary";
        if (opposite_cell_uid!=NULL_ITEM_ID)
          ostr() << " opposite " << opposite_cell_uid;
        ostr() << '\n';
        ++index;
      }
    }
    info() << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
