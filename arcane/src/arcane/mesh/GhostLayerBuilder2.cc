﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerBuilder2.cc                                       (C) 2000-2021 */
/*                                                                           */
/* Construction des couches fantomes.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/parallel/BitonicSortT.H"

#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/ISerializer.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Timer.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/IItemFamilyPolicyMng.h"
#include "arcane/IItemFamilySerializer.h"
#include "arcane/ParallelMngUtils.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

#include <algorithm>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des couches fantômes.
 */
class GhostLayerBuilder2
: public TraceAccessor
{
  class BoundaryNodeInfo;
  class BoundaryNodeBitonicSortTraits;
  class BoundaryNodeToSendInfo;

 public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;
  typedef HashTableMapT<Int32,SharedArray<Int32> > SubDomainItemMap;
  
 public:

  //! Construit une instance pour le maillage \a mesh
  GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder,bool is_allocate);
  virtual ~GhostLayerBuilder2();

 public:

  void addGhostLayers();

 private:

  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;
  IParallelMng* m_parallel_mng;
  bool m_is_verbose;
  bool m_is_allocate;

 private:
  
  void _printItem(ItemInternal* ii,ostream& o);
  void _markBoundaryItems();
  void _sendAndReceiveCells(SubDomainItemMap& cells_to_send);
  void _sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list);
  void _addGhostLayer(Integer current_layer,Int32ConstArrayView node_layer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder2::
GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder,bool is_allocate)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
, m_parallel_mng(m_mesh->parallelMng())
, m_is_verbose(false)
, m_is_allocate(is_allocate)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder2::
~GhostLayerBuilder2()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_printItem(ItemInternal* ii,ostream& o)
{
  o << ItemPrinter(ii);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GhostLayerBuilder2::BoundaryNodeInfo
{
 public:
  BoundaryNodeInfo()
  : node_uid(NULL_ITEM_UNIQUE_ID), cell_uid(NULL_ITEM_UNIQUE_ID),cell_owner(-1){}
 public:
  Int64 node_uid;
  Int64 cell_uid;
  Int32 cell_owner;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour trier les BoundaryNodeInfo via le tri bitonic.
 */
class GhostLayerBuilder2::BoundaryNodeBitonicSortTraits
{
 public:
  static bool compareLess(const BoundaryNodeInfo& k1,const BoundaryNodeInfo& k2)
  {
    Int64 k1_node_uid = k1.node_uid;
    Int64 k2_node_uid = k2.node_uid;
    if (k1_node_uid<k2_node_uid)
      return true;
    if (k1_node_uid>k2_node_uid)
      return false;

    Int64 k1_cell_uid = k1.cell_uid;
    Int64 k2_cell_uid = k2.cell_uid;
    if (k1_cell_uid<k2_cell_uid)
      return true;
    if (k1_cell_uid>k2_cell_uid)
      return false;

    return (k1.cell_owner<k2.cell_owner);
  }

  static Parallel::Request send(IParallelMng* pm,Int32 rank,ConstArrayView<BoundaryNodeInfo> values)
  {
    const BoundaryNodeInfo* fsi_base = values.data();
    return pm->send(ByteConstArrayView(messageSize(values),(const Byte*)fsi_base),rank,false);
  }

  static Parallel::Request recv(IParallelMng* pm,Int32 rank,ArrayView<BoundaryNodeInfo> values)
  {
    BoundaryNodeInfo* fsi_base = values.data();
    return pm->recv(ByteArrayView(messageSize(values),(Byte*)fsi_base),rank,false);
  }

  static Integer messageSize(ConstArrayView<BoundaryNodeInfo> values)
  {
    return CheckedConvert::toInteger(values.size()*sizeof(BoundaryNodeInfo));
  }

  static BoundaryNodeInfo maxValue()
  {
    BoundaryNodeInfo bni;
    bni.node_uid = INT64_MAX;
    bni.cell_uid = INT64_MAX;
    bni.cell_owner = -1;
    return bni;
  }

  static bool isValid(const BoundaryNodeInfo& bni)
  {
    return bni.node_uid!=INT64_MAX;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GhostLayerBuilder2::BoundaryNodeToSendInfo
{
 public:
  Integer m_index;
  Integer m_nb_cell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute les couches de mailles fantomes.
 *
 * Cette version utilise un tri pour déterminer les infos 
 *
 * Avant appel à cette fonction, il ne faut plus qu'il y ait de mailles
 * fantômes: toutes les mailles du maillages doivent appartenir à ce sous-domaine.
 * (TODO: ajouter test pour cela).
 *
 * Si on demande plusieurs couches de mailles fantômes, on procède en plusieurs
 * étapes pour le même algo. D'abord on envoie la première couche, puis la première
 * et la seconde, puis trois couches et ainsi de suite. Cela n'est probablement
 * pas optimum en terme de communication mais permet de traiter tous les cas,
 * notamment le cas ou il faut traverser plusieurs sous-domaines pour
 * ajouter des couches de mailles fantômes.
 * 
 * \todo: faire les optimisations spécifiées dans les commentaires
 * dans cette fonction.
 * \todo: faire en sorte qu'on ne travaille que avec la connectivité
 * maille/noeud.
 * 
 */ 
void GhostLayerBuilder2::
addGhostLayers()
{
  IParallelMng* pm = m_parallel_mng;
  if (!pm->isParallel())
    return;
  Integer nb_ghost_layer = m_mesh->ghostLayerMng()->nbGhostLayer();
  info() << "** GHOST LAYER BUILDER V3 With sort (nb_ghost_layer=" << nb_ghost_layer << ")";
  if (nb_ghost_layer==0) return;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  info() << " RANK="<< my_rank << " size=" << nb_rank;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  // Marque les noeuds frontières
  _markBoundaryItems();

  Integer boundary_nodes_uid_count = 0;

  // Couche fantôme à laquelle appartient le noeud.
  UniqueArray<Integer> node_layer(m_mesh->nodeFamily()->maxLocalId(),-1);
  // Couche fantôme à laquelle appartient la maille. 
  UniqueArray<Integer> cell_layer(m_mesh->cellFamily()->maxLocalId(),-1);

  // Parcours les noeuds et calcule le nombre de noeud frontières
  // et marque la première couche
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,nodes_map){
    ItemInternal* node = iid->value();
    Int32 f = node->flags();
    if (f & ItemInternal::II_Shared){
      node_layer[node->localId()] = 1;
      ++boundary_nodes_uid_count;
    }
  }

  info() << "NB BOUNDARY NODE=" << boundary_nodes_uid_count;

  for(Integer current_layer=1; current_layer<=nb_ghost_layer; ++current_layer){
    //Integer current_layer = 1;
    info() << "Processing layer " << current_layer;
    ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,cells_map){
      Cell cell = iid->value();
      //Int64 cell_uid = cell->uniqueId();
      Int32 cell_lid = cell.localId();
      if (cell_layer[cell_lid]!=(-1))
        continue;
      bool is_current_layer = false;
      for( Int32 inode_local_id : cell.nodes().localIds().range() ){
        Integer layer = node_layer[inode_local_id];
        //info() << "NODE_LAYER lid=" << i_node->localId() << " layer=" << layer;
        if (layer==current_layer){
          is_current_layer = true;
          break;
        }
      }
      if (is_current_layer){
        cell_layer[cell_lid] = current_layer;
        //info() << "Current layer celluid=" << cell_uid;
        // Si non marqué, initialise à la couche courante + 1.
        for( Int32 inode_local_id : cell.nodes().localIds().range() ){
          Integer layer = node_layer[inode_local_id];
          if (layer==(-1)){
            //info() << "Marque node uid=" << i_node->uniqueId();
            node_layer[inode_local_id] = current_layer + 1;
          }
        }
      }
    }
  }

  for( Integer i=1; i<=nb_ghost_layer; ++i )
    _addGhostLayer(i,node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_addGhostLayer(Integer current_layer,Int32ConstArrayView node_layer)
{
  info() << "Processing layer " << current_layer;

  SharedArray<BoundaryNodeInfo> boundary_node_list;
  //boundary_node_list.reserve(boundary_nodes_uid_count);

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  // On doit envoyer tous les noeuds dont le numéro de couche est différent de (-1).
  // NOTE: pour la couche au dessus de 1, il ne faut envoyer qu'une seule valeur.
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,cells_map){
    Cell cell = iid->value();
    Int64 cell_uid = cell->uniqueId();
    for( Node node : cell.nodes() ){
      Int32 node_lid = node.localId();
      bool do_it = false;
      //if (node_lid>=node_layer.size())
      //do_it = true;
      if (cell->owner()!=my_rank){
        do_it = true;
      }
      else{
        Integer layer = node_layer[node_lid];
        do_it = layer<=current_layer;
      }
      if (do_it){
        Int64 node_uid = node.uniqueId();
        BoundaryNodeInfo nci;
        nci.node_uid = node_uid;
        nci.cell_uid = cell_uid;
        nci.cell_owner = my_rank;
        boundary_node_list.add(nci);
      }
    }
  }
  info() << "NB BOUNDARY NODE LIST=" << boundary_node_list.size();

  _sortBoundaryNodeList(boundary_node_list);
  SharedArray<BoundaryNodeInfo> all_boundary_node_info = boundary_node_list;

  UniqueArray<BoundaryNodeToSendInfo> node_list_to_send;
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer bi_n = all_bni.size();
    for( Integer i=0; i<bi_n; ++i ){
      const BoundaryNodeInfo& bni = all_bni[i];
      // Recherche tous les éléments de all_bni qui ont le même noeud.
      // Cela représente toutes les mailles connectées à ce noeud.
      Int64 node_uid = bni.node_uid;
      Integer last_i = i;
      for( ; last_i<bi_n; ++last_i )
        if (all_bni[last_i].node_uid!=node_uid)
          break;
      Integer nb_same_node = (last_i - i);
      if (is_verbose)
        info() << "NB_SAME_NODE uid=" << node_uid << " n=" << nb_same_node << " last_i=" << last_i;
      // Maintenant, regarde si les mailles connectées à ce noeud ont le même propriétaire.
      // Si c'est le cas, il s'agit d'un vrai noeud frontière et il n'y a donc rien à faire.
      // Sinon, il faudra envoyer la liste des mailles à tous les PE dont les rangs apparaissent dans cette liste
      Int32 owner = bni.cell_owner;
      bool has_ghost = false;
      for( Integer z=0; z<nb_same_node; ++z )
        if (all_bni[i+z].cell_owner!=owner){
          has_ghost = true;
          break;
        }
      if (has_ghost){
        BoundaryNodeToSendInfo si;
        si.m_index = i;
        si.m_nb_cell = nb_same_node;
        node_list_to_send.add(si);
      }
      i = last_i-1;
    }
  }

  IntegerUniqueArray nb_info_to_send(nb_rank,0);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for( Integer i=0; i<nb_node_to_send; ++i ){
      Integer index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;

      ranks_done.clear();

      for( Integer kz=0; kz<nb_cell; ++kz ){
        Int32 krank = all_bni[index+kz].cell_owner;
        if (ranks_done.find(krank)==ranks_done.end()){
          ranks_done.insert(krank);
          // Pour chacun, il faudra envoyer
          // - le nombre de mailles (1*Int64)
          // - le uid du noeud (1*Int64)
          // - le uid et le rank de chaque maille (2*Int64*nb_cell)
          //TODO: il est possible de stocker les rangs sur Int32
          nb_info_to_send[krank] += (nb_cell*2) + 2;
        }
      }
    }
  }

  if (is_verbose){
    for( Integer i=0; i<nb_rank; ++i ){
      Integer nb_to_send = nb_info_to_send[i];
      if (nb_to_send!=0)
        info() << "NB_TO_SEND rank=" << i << " n=" << nb_to_send;
    }
  }

  Integer total_nb_to_send = 0;
  IntegerUniqueArray nb_info_to_send_indexes(nb_rank,0);
  for( Integer i=0; i<nb_rank; ++i ){
    nb_info_to_send_indexes[i] = total_nb_to_send;
    total_nb_to_send += nb_info_to_send[i];
  }
  info() << "TOTAL_NB_TO_SEND=" << total_nb_to_send;

  UniqueArray<Int64> resend_infos(total_nb_to_send);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for( Integer i=0; i<nb_node_to_send; ++i ){
      Integer node_index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;
      Int64 node_uid = all_bni[node_index].node_uid;

      ranks_done.clear();

      for( Integer kz=0; kz<nb_cell; ++kz ){
        Int32 krank = all_bni[node_index+kz].cell_owner;
        if (ranks_done.find(krank)==ranks_done.end()){
          ranks_done.insert(krank);
          Integer send_index =  nb_info_to_send_indexes[krank];
          resend_infos[send_index] = node_uid;
          ++send_index;
          resend_infos[send_index] = nb_cell;
          ++send_index;
          for( Integer zz=0; zz<nb_cell; ++zz ){
            resend_infos[send_index] = all_bni[node_index+zz].cell_uid;
            ++send_index;
            resend_infos[send_index] = all_bni[node_index+zz].cell_owner;
            ++send_index;
          }
          nb_info_to_send_indexes[krank] = send_index;
        }
      }
    }
  }

  IntegerUniqueArray nb_info_to_recv(nb_rank,0);
  {
    Timer::SimplePrinter sp(traceMng(),"Sending size with AllToAll");
    pm->allToAll(nb_info_to_send,nb_info_to_recv,1);
  }

  if (is_verbose)
    for( Integer i=0; i<nb_rank; ++i )
      info() << "NB_TO_RECV: I=" << i << " n=" << nb_info_to_recv[i];

  Integer total_nb_to_recv = 0;
  for( Integer i=0; i<nb_rank; ++i )
    total_nb_to_recv +=  nb_info_to_recv[i];

  // Il y a de fortes chances que cela ne marche pas si le tableau est trop grand,
  // il faut proceder avec des tableaux qui ne depassent pas 2Go a cause des
  // Int32 de MPI.
  // TODO: Faire le AllToAll en plusieurs fois si besoin.
  // TOOD: Fusionner ce code avec celui de FaceUniqueIdBuilder2.
  UniqueArray<Int64> recv_infos;
  {
    Int32 vsize = sizeof(Int64) / sizeof(Int64);
    Int32UniqueArray send_counts(nb_rank);
    Int32UniqueArray send_indexes(nb_rank);
    Int32UniqueArray recv_counts(nb_rank);
    Int32UniqueArray recv_indexes(nb_rank);
    Int32 total_send = 0;
    Int32 total_recv = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      send_counts[i] = (Int32)(nb_info_to_send[i] * vsize);
      recv_counts[i] = (Int32)(nb_info_to_recv[i] * vsize);
      send_indexes[i] = total_send;
      recv_indexes[i] = total_recv;
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    recv_infos.resize(total_nb_to_recv);

    Int64ConstArrayView send_buf(total_nb_to_send*vsize,(Int64*)resend_infos.data());
    Int64ArrayView recv_buf(total_nb_to_recv*vsize,(Int64*)recv_infos.data());

    info() << "BUF_SIZES: send=" << send_buf.size() << " recv=" << recv_buf.size();
    {
      Timer::SimplePrinter sp(traceMng(),"Send values with AllToAll");
      pm->allToAllVariable(send_buf,send_counts,send_indexes,recv_buf,recv_counts,recv_indexes);
    }
  }

  SubDomainItemMap cells_to_send(50,true);

  // TODO: il n'y a a priori pas besoin d'avoir les mailles ici mais
  // seulement la liste des procs a qui il faut envoyer. Ensuite,
  // si le proc connait a qui il doit envoyer, il peut envoyer les mailles
  // à ce moment la. Cela permet d'envoyer moins d'infos dans le AllToAll précédent

  {
    Integer index = 0;
    Int32UniqueArray my_cells;
    SharedArray<Int32> ranks_to_send;
    std::set<Int32> ranks_done;
    while (index<total_nb_to_recv){
      Int64 node_uid = recv_infos[index];
      ++index;
      Int64 nb_cell = recv_infos[index];
      ++index;
      if (is_verbose)
        info() << "NODE uid=" << node_uid << " nb_cell=" << nb_cell << " idx=" << (index-2);
      my_cells.clear();
      ranks_to_send.clear();
      ranks_done.clear();
      for( Integer kk=0; kk<nb_cell; ++kk ){
        Int64 cell_uid = recv_infos[index];
        ++index;
        Int32 cell_owner = CheckedConvert::toInt32(recv_infos[index]);
        ++index;
        if (kk==0 && current_layer==1 && m_is_allocate)
          // Je suis la maille de plus petit uid et donc je
          // positionne le propriétaire du noeud.
          // TODO: ne pas faire cela ici, mais le faire dans une routine à part.
          nodes_map[node_uid]->setOwner(cell_owner,my_rank);
        if (is_verbose)
          info() << " CELL=" << cell_uid << " owner=" << cell_owner;
        if (cell_owner==my_rank){
          ItemInternalMap::Data* dcell = cells_map.lookup(cell_uid);
          if (!dcell)
            throw FatalErrorException(A_FUNCINFO,"Internal error: cell not in our mesh");
          my_cells.add(dcell->value()->localId());
        }
        else{
          if (ranks_done.find(cell_owner)==ranks_done.end()){
            ranks_to_send.add(cell_owner);
            ranks_done.insert(cell_owner);
          }
        }
      }

      if (is_verbose)
        info() << "CELLS TO SEND: node_uid=" << node_uid
               << " nb_rank=" << ranks_to_send.size()
               << " nb_cell=" << my_cells.size();

      for( Integer zrank=0, zn=ranks_to_send.size(); zrank<zn; ++zrank ){
        SubDomainItemMap::Data* d = cells_to_send.lookupAdd(ranks_to_send[zrank]);
        Int32Array& c = d->value();
        for( Integer zid=0, zid_size=my_cells.size(); zid<zid_size; ++zid ){
          // TODO: regarder si maille pas déjà présente et ne pas l'ajouter si ce n'est pas nécessaire.
          c.add(my_cells[zid]);
        }
      }
    }
  }

  info() << "GHOST V3 SERIALIZE CELLS";
  _sendAndReceiveCells(cells_to_send);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Trie parallèle de la liste des infos sur les noeuds frontières.
 *
 * Récupère en entrée une liste de noeuds frontières et la trie en parallèle
 * en s'assurant qu'après le tri les infos d'un même noeud sont sur le
 * même proc.
 */
void GhostLayerBuilder2::
_sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  bool is_verbose = m_is_verbose;

  Parallel::BitonicSort<BoundaryNodeInfo,BoundaryNodeBitonicSortTraits> boundary_node_sorter(pm);
  boundary_node_sorter.setNeedIndexAndRank(false);

  {
    Timer::SimplePrinter sp(traceMng(),"Sorting boundary nodes");
    boundary_node_sorter.sort(boundary_node_list);
  }

  if (is_verbose){
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    for( Integer i=0; i<n; ++i ){
      const BoundaryNodeInfo& bni = all_bni[i];
      info() << "NODES_KEY i=" << i
             << " node=" << bni.node_uid
             << " cell=" << bni.cell_uid
             << " rank=" << bni.cell_owner;
    }
  }

  // TODO: il n'y a pas besoin d'envoyer toutes les mailles.
  // pour déterminer le propriétaire d'un noeud, il suffit
  // que chaque PE envoie sa maille de plus petit UID.
  // Ensuite, chaque noeud a besoin de savoir la liste
  // des sous-domaines connectés pour renvoyer l'info. Chaque
  // sous-domaine en sachant cela saura a qui il doit envoyer
  // les mailles fantomes.

  {
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    // Comme un même noeud peut être présent dans la liste du proc précédent, chaque PE
    // (sauf le 0) envoie au proc précédent le début sa liste qui contient les même noeuds.
    
    UniqueArray<BoundaryNodeInfo> end_node_list;
    Integer begin_own_list_index = 0;
    if (n!=0 && my_rank!=0){
      if (BoundaryNodeBitonicSortTraits::isValid(all_bni[0])){
        Int64 node_uid = all_bni[0].node_uid;
        for( Integer i=0; i<n; ++i ){
          if (all_bni[i].node_uid!=node_uid){
            begin_own_list_index = i;
            break;
          }
          else
            end_node_list.add(all_bni[i]);
        }
      }
    }
    info() << "BEGIN_OWN_LIST_INDEX=" << begin_own_list_index;
    if (is_verbose){
      for( Integer k=0, kn=end_node_list.size(); k<kn; ++k )
        info() << " SEND node_uid=" << end_node_list[k].node_uid
               << " cell_uid=" << end_node_list[k].cell_uid;
    }

    UniqueArray<BoundaryNodeInfo> end_node_list_recv;

    UniqueArray<Parallel::Request> requests;
    Integer recv_message_size = 0;
    Integer send_message_size = BoundaryNodeBitonicSortTraits::messageSize(end_node_list);

    // Envoie et réceptionne d'abord les tailles.
    if (my_rank!=(nb_rank-1)){
      requests.add(pm->recv(IntegerArrayView(1,&recv_message_size),my_rank+1,false));
    }
    if (my_rank!=0){
      requests.add(pm->send(IntegerConstArrayView(1,&send_message_size),my_rank-1,false));
    }
    
    pm->waitAllRequests(requests);
    requests.clear();
    
    if (recv_message_size!=0){
      Integer message_size = CheckedConvert::toInteger(recv_message_size/sizeof(BoundaryNodeInfo));
      end_node_list_recv.resize(message_size);
      requests.add(BoundaryNodeBitonicSortTraits::recv(pm,my_rank+1,end_node_list_recv));
    }
    if (send_message_size!=0)
      requests.add(BoundaryNodeBitonicSortTraits::send(pm,my_rank-1,end_node_list));

    pm->waitAllRequests(requests);

    boundary_node_list.clear();
    boundary_node_list.addRange(all_bni.subConstView(begin_own_list_index,n-begin_own_list_index));
    boundary_node_list.addRange(end_node_list_recv);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_sendAndReceiveCells(SubDomainItemMap& cells_to_send)
{
  auto exchanger { ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  // Envoie et réceptionne les mailles fantômes
  for( SubDomainItemMap::Enumerator i_map(cells_to_send); ++i_map; ){
    Int32 sd = i_map.data()->key();
    Int32Array& items = i_map.data()->value();

    // Comme la liste par sous-domaine peut contenir plusieurs
    // fois la même maille, on trie la liste et on supprime les
    // doublons
    std::sort(std::begin(items),std::end(items));
    auto new_end = std::unique(std::begin(items),std::end(items));
    items.resize(CheckedConvert::toInteger(new_end-std::begin(items)));
    info(4) << "CELLS TO SEND SD=" << sd << " NB=" << items.size();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  for( Integer i=0, ns=exchanger->nbSender(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToSend(i);
    Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    Int32ConstArrayView items_to_send = cells_to_send[rank];
    m_mesh->serializeCells(s,items_to_send);
  }
  exchanger->processExchange();
  info(4) << "END EXCHANGE CELLS";
  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_mesh->addCells(s);
  }
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Marque les entitées au bord du sous-domaine.
 *
 * Cela suppose que les faces aient déja été marquées avec le flag II_Boundary
 * et que leur propriétaire soit correctement positionné (i.e: le même pour
 * tous les sous-domaines).
 */
void GhostLayerBuilder2::
_markBoundaryItems()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();

  const int shared_and_boundary_flags = ItemInternal::II_Shared | ItemInternal::II_SubDomainBoundary;

  // Parcours les faces et marque les noeuds, arêtes et faces frontieres
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,faces_map){
    Face face = iid->value();
    ItemInternal* face_internal = face->internal();
    bool is_sub_domain_boundary_face = false;
    if (face_internal->flags() & ItemInternal::II_Boundary){
      is_sub_domain_boundary_face = true;
    }
    else{
      if (face.nbCell()==2 && (face.cell(0).owner()!=my_rank || face.cell(1).owner()!=my_rank))
        is_sub_domain_boundary_face = true;
    }
    if (is_sub_domain_boundary_face){
      face_internal->addFlags(shared_and_boundary_flags);
      for( Item inode : face.nodes() )
        inode.internal()->addFlags(shared_and_boundary_flags);
      for( Item iedge : face.edges() )
        iedge.internal()->addFlags(shared_and_boundary_flags);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_buildGhostLayerNewVersion(DynamicMesh* mesh,bool is_allocate)
{
  GhostLayerBuilder2 glb(mesh->m_mesh_builder,is_allocate);
  glb.addGhostLayers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
