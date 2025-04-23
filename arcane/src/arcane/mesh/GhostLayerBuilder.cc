// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerBuilder.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Construction des couches fantômes.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IGhostLayerMng.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/GhostLayerBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_buildGhostLayerNewVersion(DynamicMesh* mesh,bool is_allocate,Int32 version);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_DYNAMIC_MESH
// #define ARCANE_DEBUG_DYNAMIC_MESH2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder::
GhostLayerBuilder(DynamicMeshIncrementalBuilder* mesh_builder)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder::
~GhostLayerBuilder()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder::
addGhostLayers(bool is_allocate)
{
  Real begin_time = platform::getRealTime();
  Integer version = m_mesh->ghostLayerMng()->builderVersion();
  if (version==1){
    throw NotSupportedException(A_FUNCINFO,"Version 1 is no longer supported");
  }
  else if (version==2){
    info() << "Use ghost layer builder version 2";
    _addOneGhostLayerV2();
  }
  else if (version==3 || version==4){
    info() << "Use GhostLayerBuilder with sort (version " << version << ")";
    _buildGhostLayerNewVersion(m_mesh,is_allocate,version);
  }
  else
    throw NotSupportedException(A_FUNCINFO,"Bad version number for addGhostLayer");

  Real end_time = platform::getRealTime();
  Real diff = (Real)(end_time - begin_time);
  info() << "TIME to compute ghost layer=" << diff;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeCellList
{
 public:
 private:

 public:
  NodeCellList() : m_cell_last_index(5000,true) {}

 public:

  void add(Int64 node_uid,Int64 cell_uid,Int64 cell_owner)
  {
    Int32 current_index = m_cell_indexes.size();
    m_cell_indexes.add(cell_uid);
    m_cell_indexes.add(cell_owner);
    bool is_add = false;
    HashTableMapT<Int64,Int32>::Data* d = m_cell_last_index.lookupAdd(node_uid,-1,is_add);
    m_cell_indexes.add(d->value());
    d->value() = current_index;
  }

 public:
  Int64UniqueArray m_cell_indexes;
  HashTableMapT<Int64,Int32> m_cell_last_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder::
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

      s->setMode(ISerializer::ModeReserve);
      s->reserveArray(infos); // Pour les elements

      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);

      s->putArray(infos);
    }
  }
  exchanger->processExchange();
  debug() << "END EXCHANGE";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder::
_addOneGhostLayerV2()
{
  info() << "** NEW GHOST LAYER BUILDER V2";
  if (m_mesh->ghostLayerMng()->nbGhostLayer()!=1)
    ARCANE_THROW(NotImplementedException,"Only one layer of ghost cells is supported");

  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  debug() << " RANK="<< pm->commRank() << " size=" << pm->commSize();
  if (!pm->isParallel()){
    debug() << "NOT PARALLEL";
    return;
  }
#ifdef ARCANE_DEBUG_DYNAMIC_MESH
  const bool is_verbose = true;
#else
  const bool is_verbose = false;
#endif

  OStringStream ostr;
  if (is_verbose)
    ostr() << "** FACES LIST\n";

  Integer nb_sub_domain_boundary_face = 0;
  // Marque les noeuds sur la frontière
  ItemInternalMap& cells_map = m_mesh->cellsMap(); // Supporte les transferts du maillage
  ItemInternalMap& faces_map = m_mesh->facesMap(); // Détermine les frontières avant transfert
  // ItemInternalMap& edges_map = m_mesh->edgesMap(); // N'est pas utilisé directement par l'algo
  ItemInternalMap& nodes_map = m_mesh->nodesMap(); // Localise les modifications

  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;
  // Parcours les faces et marque les nœuds, arêtes et faces frontières
  faces_map.eachItem([&](Face face) {
    impl::ItemBase face_base = face.itemBase();
    if (is_verbose){
      ostr() << ItemPrinter(face);
      ostr() << '\n';
    }
    bool is_sub_domain_boundary_face = false;
    if (face_base.hasFlags(ItemFlags::II_Boundary)) {
      is_sub_domain_boundary_face = true;
    }
    else{
      if (face.nbCell()==2 && (face.cell(0).owner()!=my_rank || face.cell(1).owner()!=my_rank))
        is_sub_domain_boundary_face = true;
    }
    if (is_sub_domain_boundary_face){
      face_base.toMutable().addFlags(shared_and_boundary_flags);
      ++nb_sub_domain_boundary_face;
      for( Item inode : face.nodes() )
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
      for( Item iedge : face.edges() )
        iedge.mutableItemBase().addFlags(shared_and_boundary_flags);
    }
  });

  Integer boundary_nodes_uid_count = 0;

  // Parcours les nœuds et ajoute les nœuds frontières
  Int64 my_max_node_uid = NULL_ITEM_UNIQUE_ID;
  nodes_map.eachItem([&](Node node) {
    Int32 f = node.itemBase().flags();
    if (f & ItemFlags::II_Shared) {
      Int64 node_uid = node.uniqueId();
      if (node_uid > my_max_node_uid)
        my_max_node_uid = node_uid;
      ++boundary_nodes_uid_count;
    }
  });

  Int64 global_max_node_uid = pm->reduce(Parallel::ReduceMax,my_max_node_uid);
  debug() << "NB BOUNDARY NODE=" << boundary_nodes_uid_count
         << " MY_MAX_UID=" << my_max_node_uid
         << " GLOBAL=" << global_max_node_uid;

  if (is_verbose){
    ostr.reset();
    ostr() << "List of shared cells:\n";
  }


  //TODO: choisir bonne valeur pour initialiser la table
  BoundaryInfosMap boundary_infos_to_send(200,true);
  NodeUidToSubDomain uid_to_subdomain_converter(global_max_node_uid,nb_rank);

  cells_map.eachItem([&](Cell cell) {
    if (is_verbose){
      ostr() << "Send cell " << ItemPrinter(cell) << '\n';
    }
    //info() << " CHECK cell uid=" << cell->uniqueId() << " owner=" << cell->owner();
    //bool add_cell = false;
    for( Node node : cell.nodes() ){
      //info() << "** CHECK NODE node=" << i_node->uniqueId() << " cell=" << cell->uniqueId();
      if (node.hasFlags(ItemFlags::II_Shared)){
        Int64 node_uid = node.uniqueId();
        //info() << "** ADD BOUNDARY CELL node=" << node_uid << " cell=" << cell->uniqueId();
        Int32 dest_rank = uid_to_subdomain_converter.uidToRank(node_uid);
        SharedArray<Int64> v = boundary_infos_to_send.lookupAdd(dest_rank)->value();
        v.add(node_uid);
        v.add(cell.owner());
        v.add(cell.uniqueId());
        //TODO: supprimer les doublons ?
        //add_cell = true;
        //break;
      }
    }
  });

  if (is_verbose)
    info() << ostr.str();

  info() << "Number of shared faces: " << nb_sub_domain_boundary_face;

  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };

  if (!platform::getEnvironmentVariable("ARCANE_COLLECTIVE_GHOST_LAYER").null())
    exchanger->setExchangeMode(IParallelExchanger::EM_Collective);

  _exchangeData(exchanger.get(),boundary_infos_to_send);

  traceMng()->flush();
  pm->barrier();
  NodeCellList node_cell_list;
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      s->getArray(received_infos);
      Int64 nb_info = received_infos.largeSize();
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      if ((nb_info % 3)!=0)
        ARCANE_FATAL("Inconsistent received data v={0}",nb_info);
      Int64 nb_info_true = nb_info / 3;
      for( Int64 z=0; z<nb_info_true; ++z ){
        Int64 node_uid = received_infos[(z*3)+0];
        Int64 cell_owner = received_infos[(z*3)+1];
        Int64 cell_uid = received_infos[(z*3)+2];
        node_cell_list.add(node_uid,cell_uid,cell_owner);
      }
    }
  }

  boundary_infos_to_send = BoundaryInfosMap(1000,true);

  {
    Int64ConstArrayView cell_indexes = node_cell_list.m_cell_indexes;
    debug() << "NB_CELL_INDEXES = " << cell_indexes.size();
    //for( Integer i=0, s=cell_indexes.size(); i<s; ++i )
    //info() << "INDEX I=" << i << " V=" << cell_indexes[i];
    Int32UniqueArray ranks;
    Int64UniqueArray cells;
    for( HashTableMapEnumeratorT<Int64,Int32> i_map(node_cell_list.m_cell_last_index); ++i_map; ){
      HashTableMapT<Int64,Int32>::Data* d = i_map.data();
      Int32 index = d->value();
      Int64 node_uid = d->key();
      //info() << "NODE UID=" << node_uid;
      ranks.clear();
      cells.clear();
      // Comme on connait la liste des mailles connectées à ce noeud ainsi que leur
      // propriétaire, en profite pour calculer le propriètaire du noeud en considérant
      // qu'il s'agit du même propriétaire que celui de la maille de plus petit uniqueId()
      // connecté à ce noeud.
      Int32 node_new_owner = NULL_SUB_DOMAIN_ID;
      Int64 smallest_cell_uid = NULL_ITEM_UNIQUE_ID;
      //TODO ajouter securite en calculant le nombre max de valeurs
      while(index!=(-1)){
        Int64 cell_uid = cell_indexes[index];
        Int32 cell_owner = CheckedConvert::toInt32(cell_indexes[index+1]);
        index = CheckedConvert::toInteger(cell_indexes[index+2]);
        //info() << " CELLS: uid=" << cell_uid << " owner=" << cell_owner;
        ranks.add((Int32)cell_owner);
        cells.add(cell_uid);
        if (cell_uid<smallest_cell_uid || node_new_owner==NULL_SUB_DOMAIN_ID){
          smallest_cell_uid = cell_uid;
          node_new_owner = cell_owner;
        }
      }
      // Tri les rangs puis supprime les doublons
      std::sort(std::begin(ranks),std::end(ranks));
      Integer new_size = CheckedConvert::toInteger(std::unique(std::begin(ranks),std::end(ranks)) - std::begin(ranks));
      ranks.resize(new_size);
      //info() << "NEW_SIZE=" << new_size;
      //for( Integer z=0; z<new_size; ++z )
      //info() << "NEW_RANK=" << ranks[z];

      // Si le nombre de rang vaut 1, cela signifie que le noeud n'appartient qu'à un seul sous-domaine
      // et donc il s'agit d'un vrai noeud frontière. Il n'y a pas besoin de transférer ses mailles.
      if (new_size==1)
        continue;
      Integer nb_cell = cells.size();
      for( Integer z=0; z<new_size; ++z ){
        Int32 dest_rank = ranks[z];
        //info() << "NEW_RANK=" << dest_rank;
        Int64Array& v = boundary_infos_to_send.lookupAdd(dest_rank)->value();
        v.add(node_uid);
        v.add(node_new_owner);
        v.add(new_size);
        v.add(nb_cell);
        for( Integer z2=0; z2<new_size; ++z2 )
          v.add(ranks[z2]);
        for( Integer z2=0; z2<nb_cell; ++z2 )
          v.add(cells[z2]);
      }
    }
  }

  exchanger = ParallelMngUtils::createExchangerRef(pm);
  _exchangeData(exchanger.get(),boundary_infos_to_send);
  debug() << "END OF EXCHANGE";

  typedef HashTableMapT<Int32,SharedArray<Int32> > SubDomainItemMap;
  SubDomainItemMap cells_to_send(50,true);
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER 2 =" << nb_receiver;
    Int64UniqueArray received_infos;
    //HashTableMapT<Int64,Int32> nodes_nb_cell(1000,true);
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      s->getArray(received_infos);
      Int64 nb_info = received_infos.largeSize();
      Int64 z = 0;
      Int32UniqueArray ranks;
      Int32UniqueArray cells;
      while (z<nb_info){
        Int64 node_uid = received_infos[z];
        Int32 node_new_owner = CheckedConvert::toInt32(received_infos[z+1]);
        Int32 nb_rank = CheckedConvert::toInt32(received_infos[z+2]);
        Int32 nb_cell = CheckedConvert::toInt32(received_infos[z+3]);
        //info() << "RECEIVE NODE uid="<< node_uid << " nb_rank=" << nb_rank << " nb_cell=" << nb_cell;
        nodes_map.findItem(node_uid).toMutable().setOwner(node_new_owner, my_rank);
        ranks.clear();
        cells.clear();
        z += 4;
        for( Integer z2=0; z2<nb_rank; ++z2 ){
          Int32 nrank = (Int32)received_infos[z+z2];
          if (nrank!=my_rank)
            ranks.add(nrank);
        }
        z += nb_rank;
        for( Integer z2=0; z2<nb_cell; ++z2 ){
          Int64 cell_uid = received_infos[z+z2];
          impl::ItemBase dcell = cells_map.tryFind(cell_uid);
          if (!dcell.null())
            cells.add(dcell.localId());
        }
        for( Integer z2=0,zs=ranks.size(); z2<zs; ++z2 ){
          SubDomainItemMap::Data* d = cells_to_send.lookupAdd(ranks[z2]);
          SharedArray<Int32> dv = d->value();
          for( Integer z3=0, zs3=cells.size(); z3<zs3; ++z3 )
            dv.add(cells[z3]);
        }
        z += nb_cell;
      }
    }
  }

  // Envoie et réceptionne les mailles fantômes
  _exchangeCells(cells_to_send,false);
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder::
_exchangeCells(HashTableMapT<Int32,SharedArray<Int32>>& cells_to_send,bool with_flags)
{
  //TODO: fusionner avec GhostLayerBuilder2::_exchangeCells().
  typedef HashTableMapT<Int32,SharedArray<Int32>> SubDomainItemMap;
  IParallelMng* pm = m_mesh->parallelMng();
  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };
  for( SubDomainItemMap::Enumerator i_map(cells_to_send); ++i_map; ){
    Int32 sd = i_map.data()->key();
    // TODO: items peut contenir des doublons et donc il faudrait les supprimer
    // pour éviter d'envoyer inutilement plusieurs fois la même maille.
    Int32ConstArrayView items = i_map.data()->value();
    info(4) << "CELLS TO SEND SD=" << sd << " NB=" << items.size();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  for( Integer i=0, ns=exchanger->nbSender(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToSend(i);
    Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    Int32ConstArrayView items_to_send = cells_to_send[rank];
    //m_mesh->serializeCells(s,items_to_send,with_flags);
    ScopedPtrT<IItemFamilySerializer> cell_serializer(m_mesh->cellFamily()->policyMng()->createSerializer(with_flags));
    s->setMode(ISerializer::ModeReserve);
    cell_serializer->serializeItems(s,items_to_send);
    s->allocateBuffer();
    s->setMode(ISerializer::ModePut);
    cell_serializer->serializeItems(s,items_to_send);
  }
  exchanger->processExchange();
  info(4) << "END EXCHANGE CELLS";
  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    //m_mesh->addCells(s,with_flags);
    s->setMode(ISerializer::ModeGet);
    ScopedPtrT<IItemFamilySerializer> cell_serializer(m_mesh->cellFamily()->policyMng()->createSerializer(with_flags));
    cell_serializer->deserializeItems(s,nullptr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder::
addGhostChildFromParent()
{
  info() << "** AMR GHOST CHILD FROM PARENT BUILDER V1";

  IParallelMng* pm = m_mesh->parallelMng();
  debug() << " RANK="<< pm->commRank() << " size=" << pm->commSize();
  if (!pm->isParallel()){
    debug() << "NOT PARALLEL";
    return;
  }
  Integer sid = pm->commRank();

  // Marque les noeuds sur la frontière
  ItemInternalMap& cells_map = m_mesh->cellsMap();

  FaceFamily& true_face_family = m_mesh->trueFaceFamily();

  //TODO: choisir bonne valeur pour initialiser la table
  BoundaryInfosMap boundary_infos_to_send(200,true);

  cells_map.eachItem([&](Item cell) {
    ARCANE_ASSERT((cell.owner() != -1), (""));
    if (cell.itemBase().level() == 0 && cell.owner() != sid) {
      ARCANE_ASSERT((cell.owner() != -1), (""));
      Int64Array& v = boundary_infos_to_send.lookupAdd(cell.owner())->value();
      v.add(sid);
      v.add(cell.uniqueId());
    }
  });

  // Positionne la liste des envois
  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };
  _exchangeData(exchanger.get(),boundary_infos_to_send);

  traceMng()->flush();
  pm->barrier();

  typedef HashTableMapT<Int32,SharedArray<Int32> > SubDomainItemMap;
  SubDomainItemMap cells_to_send(50,true);
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      s->getArray(received_infos);
      Int64 nb_info = received_infos.size();
      //Int32 orig_rank = sm->destSubDomain();
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      if ((nb_info % 2)!=0)
        ARCANE_FATAL("info size can not be divided by 2 v={0}",nb_info);
      Int64 nb_info_true = nb_info / 2;
      Integer nb_recv_child=0;
      for( Int64 z=0; z<nb_info_true; ++z ){
        Int32 cell_owner = CheckedConvert::toInt32(received_infos[(z*2)+0]);
        Int64 cell_uid = received_infos[(z*2)+1];

        impl::ItemBase cell = cells_map.findItem(cell_uid);
        ARCANE_ASSERT((cell.uniqueId() == cell_uid), (""));
        if (!cell.hasHChildren())
          continue;
        UniqueArray<ItemInternal*> cell_family;
        ARCANE_ASSERT((cell.level() == 0), (""));
        ARCANE_ASSERT((cell.owner() != -1), ("CELL"));
        ARCANE_ASSERT((cell_owner != -1),("CELL"));
        true_face_family.familyTree(cell_family,cell);
        SubDomainItemMap::Data* d = cells_to_send.lookupAdd(cell_owner);
        Int32Array& dv = d->value();
        const Integer cs=cell_family.size();
        nb_recv_child +=cs;
        for(Integer c=1;c<cs;c++){
          ItemInternal* child= cell_family[c];
          ARCANE_ASSERT((child->owner() != -1),("CHILD"));
          //debug() << child->topHParent()->uniqueId() << " " << cell->uniqueId() << " " << child->topHParent()->owner() << " " << cell->owner();
          //ARCANE_ASSERT((child->topHParent() == cell),("CHILD"));
          dv.add(child->localId());
        }
        cell_family.clear();
      }
      debug() << "nb_recv_child= " << nb_recv_child;
    }
  }
  // Envoie et réceptionne les mailles fantômes
  _exchangeCells(cells_to_send,true);
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void GhostLayerBuilder::
addGhostChildFromParent2(Array<Int64>& ghost_cell_to_refine)
{
  info() << "** AMR GHOST CHILD FROM PARENT BUILDER V2";

  IParallelMng* pm = m_mesh->parallelMng();
  debug() << " RANK="<< pm->commRank() << " size=" << pm->commSize();
  if (!pm->isParallel()){
    debug() << "NOT PARALLEL";
    return;
  }
  Integer sid = pm->commRank();

  // Marque les noeuds sur la frontière
  ItemInternalMap& cells_map = m_mesh->cellsMap();

  //TODO: choisir bonne valeur pour initialiser la table
  BoundaryInfosMap boundary_infos_to_send(200,true);
  // mailles de niveau 0 ne sont pas concernée
  // que les mailles actives de niveau supérieur a 0 sont concernees
  // que les mailles qui viennent d'être raffinées ou dé-raffinées sont concernées
  cells_map.eachItem([&](Item cell) {
    ARCANE_ASSERT((cell.owner() != -1), (""));
    if (cell.owner() == sid)
      return;
    // cela suppose que les flags sont deja synchronises
    if (cell.hasFlags(ItemFlags::II_JustRefined)) {
      // cell to add
      ghost_cell_to_refine.add(cell.uniqueId());
      Int64Array& v = boundary_infos_to_send.lookupAdd(cell.owner())->value();
      v.add(sid);
      v.add(cell.uniqueId());
    }
  });

  // Positionne la liste des envois
  auto exchanger{ ParallelMngUtils::createExchangerRef(pm) };
  _exchangeData(exchanger.get(), boundary_infos_to_send);

  traceMng()->flush();
  pm->barrier();

  typedef HashTableMapT<Int32,SharedArray<Int32> > SubDomainItemMap;
  SubDomainItemMap cells_to_send(50,true);
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      s->getArray(received_infos);
      Int64 nb_info = received_infos.size();
      if ((nb_info % 2)!=0)
        ARCANE_FATAL("info size can not be divided by 2 v={0}",nb_info);
      Int64 nb_info_true = nb_info / 2;
      Integer nb_recv_child = 0;
      for( Int64 z=0; z<nb_info_true; ++z ){
        Int32 cell_owner = CheckedConvert::toInt32(received_infos[(z*2)+0]);
        Int64 cell_uid = received_infos[(z*2)+1];

        Cell cell = cells_map.findItem(cell_uid);
        ARCANE_ASSERT((cell.uniqueId() == cell_uid),(""));
        ARCANE_ASSERT((cell.owner() != -1),("CELL"));
        ARCANE_ASSERT((cell_owner != -1),("CELL"));

        SubDomainItemMap::Data* d = cells_to_send.lookupAdd(cell_owner);
        Int32Array& dv = d->value();

        nb_recv_child +=cell.nbHChildren();
        for(Integer c=0,cs=cell.nbHChildren();c<cs;c++){
          Cell child= cell.hChild(c);
          ARCANE_ASSERT((child.owner() != -1),("CHILD"));
          //debug() << child->topHParent()->uniqueId() << " " << cell->uniqueId() << " " << child->topHParent()->owner() << " " << cell->owner();
          //ARCANE_ASSERT((child->topHParent() == cell),("CHILD"));
          dv.add(child.localId());
        }
      }
      debug() << "nb_recv_child= " << nb_recv_child;
    }
  }

  // Envoie et réceptionne les mailles fantômes
  _exchangeCells(cells_to_send,true);
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeUidToSubDomain::
NodeUidToSubDomain(Int64 max_uid,Int32 nb_rank)
: m_nb_rank(nb_rank)
, m_modulo(1)
, m_nb_by_rank(max_uid)
{
  m_nb_by_rank = max_uid / nb_rank;
  if (m_nb_by_rank==0)
    m_nb_by_rank = max_uid;
  m_modulo = nb_rank;
  Integer div_value = 1;
  if (m_nb_rank>4)
    div_value = 2;
  String s = platform::getEnvironmentVariable("ARCANE_INIT_RANK_GROUP_SIZE");
  if (!s.null()){
    bool is_ok = builtInGetValue(div_value,s);
    if (is_ok){
      if (div_value<0)
        div_value = 1;
      if (div_value>m_nb_rank)
        div_value = m_nb_rank;
    }
  }
  m_modulo = m_nb_rank / div_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
