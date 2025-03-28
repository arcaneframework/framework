// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeUniqueIdBuilder.cc                                      (C) 2000-2024 */
/*                                                                           */
/* Construction des identifiants uniques des arêtes.                         */
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

#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IMeshUniqueIdMng.h"

#include <functional>

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeUniqueIdBuilder::
computeEdgesUniqueIds()
{
  double begin_time = platform::getRealTime();

  Int32 edge_version = m_mesh->meshUniqueIdMng()->edgeBuilderVersion();

  info() << "Using version=" << edge_version << " to compute edges unique ids"
         << " mesh=" << m_mesh->name();

  if (edge_version == 1)
    _computeEdgesUniqueIdsParallel3();
  else if (edge_version == 2)
    _computeEdgesUniqueIdsParallelV2();
  else if (edge_version == 3)
    _computeEdgesUniqueIdsParallel64bit();
  else if (edge_version == 0)
    info() << "No renumbering for edges";
  else
    ARCANE_FATAL("Invalid valid version '{0}'. Valid values are 0, 1, 2 or 3");

  double end_time = platform::getRealTime();
  Real diff = (Real)(end_time - begin_time);
  info() << "TIME to compute edge unique ids=" << diff;

  ItemInternalMap& edges_map = m_mesh->edgesMap();

  // Il faut ranger à nouveau #m_edges_map car les uniqueId() des
  // edges ont été modifiés
  edges_map.notifyUniqueIdsChanged();

  if (m_mesh_builder->isVerbose()) {
    info() << "NEW EDGES_MAP after re-indexing";
    edges_map.eachItem([&](Item edge) {
      info() << "Edge uid=" << edge.uniqueId() << " lid=" << edge.localId();
    });
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

  T_CellEdgeInfo(Int64 uid, Integer nb_back_edge, Integer nb_true_boundary_edge)
  : m_unique_id(uid)
  , m_nb_back_edge(nb_back_edge)
  , m_nb_true_boundary_edge(nb_true_boundary_edge)
  {
  }

  T_CellEdgeInfo()
  : m_unique_id(NULL_ITEM_ID)
  , m_nb_back_edge(0)
  , m_nb_true_boundary_edge(0)
  {
  }

 public:

  bool operator<(const T_CellEdgeInfo& ci) const
  {
    return m_unique_id < ci.m_unique_id;
  }

 public:

  Int64 m_unique_id;
  Int64 m_nb_back_edge;
  Int64 m_nb_true_boundary_edge;
};

template <typename DataType>
class ItemInfoMultiList
{
 public:
 private:

  class MyInfo
  {
   public:

    MyInfo(const DataType& d, Integer n)
    : data(d)
    , next_index(n)
    {}

   public:

    DataType data;
    Integer next_index;
  };

 public:

  ItemInfoMultiList()
  : m_last_index(5000, true)
  {}

 public:

  void add(Int64 node_uid, const DataType& data)
  {
    Integer current_index = m_values.size();

    bool is_add = false;
    HashTableMapT<Int64, Int32>::Data* d = m_last_index.lookupAdd(node_uid, -1, is_add);

    m_values.add(MyInfo(data, d->value()));
    d->value() = current_index;
  }

 public:

  UniqueArray<MyInfo> m_values;
  HashTableMapT<Int64, Int32> m_last_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Parallel3EdgeUniqueIdBuilder
: public TraceAccessor
{
  using BoundaryInfosMap = std::unordered_map<Int32, SharedArray<Int64>>;

 public:

  Parallel3EdgeUniqueIdBuilder(ITraceMng* tm, DynamicMeshIncrementalBuilder* mesh_builder,
                               Int64 max_node_uid);

 public:

  void compute();

 private:

  DynamicMesh* m_mesh = nullptr;
  DynamicMeshIncrementalBuilder* m_mesh_builder = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  const Int32 m_my_rank = A_NULL_RANK;
  const Int32 m_nb_rank = A_NULL_RANK;
  BoundaryInfosMap m_boundary_infos_to_send;
  NodeUidToSubDomain m_uid_to_subdomain_converter;
  std::unordered_map<Int64, SharedArray<Int64>> m_nodes_info;
  UniqueArray<bool> m_is_boundary_nodes;
  bool m_is_verbose = false;

 private:

  void _exchangeData(IParallelExchanger* exchanger);
  void _addEdgeBoundaryInfo(Edge edge);
  void _computeEdgesUniqueId();
  void _sendInfosToOtherRanks();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Parallel3EdgeUniqueIdBuilder::
Parallel3EdgeUniqueIdBuilder(ITraceMng* tm, DynamicMeshIncrementalBuilder* mesh_builder, Int64 max_node_uid)
: TraceAccessor(tm)
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
, m_parallel_mng(m_mesh->parallelMng())
, m_my_rank(m_parallel_mng->commRank())
, m_nb_rank(m_parallel_mng->commSize())
, m_uid_to_subdomain_converter(max_node_uid, m_nb_rank)
, m_is_verbose(m_mesh_builder->isVerbose())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * //COPIE DEPUIS GhostLayerBuilder.
 * Faire une classe unique.
 */
void Parallel3EdgeUniqueIdBuilder::
_exchangeData(IParallelExchanger* exchanger)
{
  for (const auto& [key, value] : m_boundary_infos_to_send) {
    exchanger->addSender(key);
  }
  exchanger->initializeCommunicationsMessages();
  {
    for (Integer i = 0, ns = exchanger->nbSender(); i < ns; ++i) {
      ISerializeMessage* sm = exchanger->messageToSend(i);
      Int32 rank = sm->destination().value();
      ISerializer* s = sm->serializer();
      ConstArrayView<Int64> infos = m_boundary_infos_to_send[rank];
      Integer nb_info = infos.size();
      s->setMode(ISerializer::ModeReserve);
      s->reserveInt64(1); // Pour le nombre d'éléments
      s->reserveSpan(eBasicDataType::Int64, nb_info); // Pour les elements
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Parallel3EdgeUniqueIdBuilder::
_addEdgeBoundaryInfo(Edge edge)
{
  Node first_node = edge.node(0);
  Int64 first_node_uid = first_node.uniqueId();
  SharedArray<Int64> v;
  Int32 dest_rank = -1;
  if (!m_is_boundary_nodes[first_node.localId()]) {
    v = m_nodes_info[first_node_uid];
  }
  else {
    dest_rank = m_uid_to_subdomain_converter.uidToRank(first_node_uid);
    v = m_boundary_infos_to_send[dest_rank];
  }
  v.add(first_node_uid); // 0
  v.add(m_my_rank); // 1
  v.add(edge.uniqueId()); // 2
  v.add(edge.type()); // 3
  v.add(NULL_ITEM_UNIQUE_ID); // 4 : only used for debug
  v.add(NULL_ITEM_UNIQUE_ID); // 5 : only used for debug
  if (m_is_verbose)
    info() << "Edge uid=" << edge.uniqueId() << " n0,n1=" << edge.node(0).uniqueId() << "," << edge.node(1).uniqueId()
           << " n0=" << ItemPrinter(edge.node(0)) << " n1=" << ItemPrinter(edge.node(1)) << " dest_rank=" << dest_rank;
  for (Node edge_node : edge.nodes())
    v.add(edge_node.uniqueId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcule les numéros uniques de chaque edge en parallèle.
  
  NEW VERSION.

  NOTE: GG Juin 2022
  Il semble que cette version ne fonctionne pas toujours lorsqu'elle est
  appelée est que le maillage est déjà découpé. Cela est du au fait que
  l'algorithme si-dessus est recopié sur celui qui calcule les uniqueId() des
  faces (dans FaceUniqueIdBuilder). Cependant, l'algorithme de calcul des faces
  suppose qu'une face frontière n'existe que dans une seule partie (un seul rang)
  ce qui n'est pas le cas pour les arêtes. On se retrouve alors avec des sous-domaines
  qui n'ont pas leurs arêtes renumérotées.
*/
void Parallel3EdgeUniqueIdBuilder::
compute()
{
  IParallelMng* pm = m_mesh->parallelMng();

  Integer nb_local_edge = m_mesh_builder->oneMeshItemAdder()->nbEdge();
  info() << "ComputeEdgesUniqueIdsParallel3 nb_edge=" << nb_local_edge;

  UniqueArray<Int64> edges_opposite_cell_uid(nb_local_edge);
  edges_opposite_cell_uid.fill(NULL_ITEM_ID);
  UniqueArray<Int32> edges_opposite_cell_index(nb_local_edge);
  UniqueArray<Int32> edges_opposite_cell_owner(nb_local_edge);

  // Pour vérification, s'assure que tous les éléments de ce tableau
  // sont valides, ce qui signifie que toutes les edges ont bien été
  // renumérotés.
  UniqueArray<Int64> edges_new_uid(nb_local_edge);
  edges_new_uid.fill(NULL_ITEM_UNIQUE_ID);

  UniqueArray<Int64> edges_infos;
  edges_infos.reserve(10000);
  ItemInternalMap& edges_map = m_mesh->edgesMap();
  ItemInternalMap& faces_map = m_mesh->facesMap(); // utilisé pour détecter le bord

  // NOTE : ce tableau n'est pas utile sur toutes les mailles. Il
  // suffit qu'il contienne les mailles dont on a besoin, c'est-à-dire
  // les nôtres + celles connectées à une de nos edges.
  HashTableMapT<Int32, Int32> cell_first_edge_uid(m_mesh_builder->oneMeshItemAdder()->nbCell() * 2, true);

  // Rassemble les données des autres processeurs dans recv_cells;
  // Pour éviter que les tableaux ne soient trop gros, on procède en plusieurs
  // étapes.
  // Chaque sous-domaine construit sa liste des arêtes frontières, avec pour
  // chaque arête :
  //  - son type,
  //  - la liste de ses noeuds,
  //  - le numéro unique de sa maille,
  //  - le propriétaire de sa maille,
  //  - son indice dans sa maille,
  // Cette liste sera ensuite envoyée à tous les sous-domaines.

  IItemFamily* node_family = m_mesh->nodeFamily();
  m_is_boundary_nodes.resize(node_family->maxLocalId(), false);

  // Marque tous les noeuds frontières, car ce sont ceux qu'il faudra envoyer
  // Un noeud est considéré comme frontière s'il appartient à une face qui n'a qu'une
  // maille connectée.
  faces_map.eachItem([&](Face face) {
    Integer face_nb_cell = face.nbCell();
    if (face_nb_cell == 1) {
      for (Int32 ilid : face.nodeIds())
        m_is_boundary_nodes[ilid] = true;
    }
  });

  // Détermine la liste des arêtes frontières.
  // L'ordre de cette liste dépend de l'implémentation de la table de hashage.
  // Afin d'avoir la même numérotation que la version historique (qui utilise HashTableMapT),
  // on utilise une instance temporaire de cette classe pour ce calcul si
  // l'implémentation utilisée est différente. C'est le cas à partir d'octobre 2024.
  // A terme, il faudrait utiliser une autre version du calcul des uniqueId() des
  // arêtes.
  // TODO: Ce mécanisme est en test. A vérifier que cela donne ensuite
  // la même numérotation.
  const bool is_new_item_map_impl = ItemInternalMap::UseNewImpl;
  if (is_new_item_map_impl) {
    info() << "Edge: ItemInternalMap is using new implementation";
    HashTableMapT<Int64, Edge> old_edges_map(5000, false);
    edges_map.eachItem([&](Edge edge) {
      old_edges_map.add(edge.uniqueId(), edge);
    });
    old_edges_map.eachValue([&](Edge edge) {
      _addEdgeBoundaryInfo(edge);
    });
  }
  else {
    edges_map.eachItem([&](Edge edge) {
      _addEdgeBoundaryInfo(edge);
    });
  }

  _computeEdgesUniqueId();
  _sendInfosToOtherRanks();

  traceMng()->flush();
  pm->barrier();
  info() << "END OF TEST NEW EDGE COMPUTE";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Parallel3EdgeUniqueIdBuilder::
_computeEdgesUniqueId()
{
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  IParallelMng* pm = m_parallel_mng;
  Ref<IParallelExchanger> exchanger{ ParallelMngUtils::createExchangerRef(pm) };
  _exchangeData(exchanger.get());

  Integer nb_receiver = exchanger->nbReceiver();
  debug() << "NB RECEIVER=" << nb_receiver;
  SharedArray<Int64> received_infos;
  for (Integer i = 0; i < nb_receiver; ++i) {
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
    Integer z = 0;
    while (z < nb_info) {
      Int64 node_uid = received_infos[z + 0];
      //Int64 sender_rank = received_infos[z+1];
      //Int64 edge_uid = received_infos[z+2];
      Int32 edge_type = (Int32)received_infos[z + 3];
      // received_infos[z+4];
      // received_infos[z+5];
      ItemTypeInfo* itt = itm->typeFromId(edge_type);
      Integer edge_nb_node = itt->nbLocalNode();
      Int64Array& a = m_nodes_info[node_uid];
      a.addRange(Int64ConstArrayView(6 + edge_nb_node, &received_infos[z]));
      z += 6;
      z += edge_nb_node;
      //info() << "NODE UID=" << node_uid << " sender=" << sender_rank
      //       << " edge_uid=" << edge_uid;
      //node_cell_list.add(node_uid,cell_uid,cell_owner);
      //HashTableMapT<Int64,Int32>::Data* v = nodes_nb_cell.lookupAdd(node_uid);
      //++v->value();
    }
  }

  Integer my_max_edge_node = 0;
  for (const auto& [key, value] : m_nodes_info) {
    //Int64 key = inode.data()->key();
    Int64ConstArrayView a = value;
    //info() << "A key=" << key << " size=" << a.size();
    Integer nb_info = a.size();
    Integer z = 0;
    Integer node_nb_edge = 0;
    while (z < nb_info) {
      ++node_nb_edge;
      //Int64 node_uid = a[z+0];
      //Int64 sender_rank = a[z+1];
      //Int64 edge_uid = a[z+2];
      Int32 edge_type = (Int32)a[z + 3];
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
    my_max_edge_node = math::max(node_nb_edge, my_max_edge_node);
  }
  Integer global_max_edge_node = pm->reduce(Parallel::ReduceMax, my_max_edge_node);
  debug() << "GLOBAL MAX EDGE NODE=" << global_max_edge_node;
  // OK, maintenant donne comme uid de la edge (node_uid * global_max_edge_node + index)
  IntegerUniqueArray indexes;
  m_boundary_infos_to_send.clear();

  for (const auto& [key, value] : m_nodes_info) {
    Int64ConstArrayView a = value;
    Integer nb_info = a.size();
    Integer z = 0;
    Integer node_nb_edge = 0;
    indexes.clear();
    while (z < nb_info) {
      Int64 node_uid = a[z + 0];
      Int32 sender_rank = (Int32)a[z + 1];
      Int64 edge_uid = a[z + 2];
      Int32 edge_type = (Int32)a[z + 3];
      // a[z+4];
      // a[z+5];
      ItemTypeInfo* itt = itm->typeFromId(edge_type);
      Integer edge_nb_node = itt->nbLocalNode();

      // Regarde si la edge est déjà dans la liste:
      Integer edge_index = node_nb_edge;
      Int32 edge_new_owner = sender_rank;
      for (Integer y = 0; y < node_nb_edge; ++y) {
        if (memcmp(&a[indexes[y] + 6], &a[z + 6], sizeof(Int64) * edge_nb_node) == 0) {
          edge_index = y;
          edge_new_owner = (Int32)a[indexes[y] + 1];
        }
      }
      Int64 edge_new_uid = (node_uid * global_max_edge_node) + edge_index;
      Int64Array& v = m_boundary_infos_to_send[sender_rank];
      // Indique au propriétaire de cette arête son nouvel uid
      v.add(edge_uid);
      v.add(edge_new_uid);
      v.add(edge_new_owner);
      indexes.add(z);
      z += 6;
      z += edge_nb_node;
      /* info() << "NODE3 UID=" << node_uid << " sender=" << sender_rank
               << " edge_uid=" << edge_uid
               << " edge_index=" << edge_index
               << " edge_new_uid=" << edge_new_uid; */
      ++node_nb_edge;
    }
    my_max_edge_node = math::max(node_nb_edge, my_max_edge_node);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Parallel3EdgeUniqueIdBuilder::
_sendInfosToOtherRanks()
{
  const bool is_verbose = m_mesh_builder->isVerbose();
  IParallelMng* pm = m_parallel_mng;
  Ref<IParallelExchanger> exchanger = ParallelMngUtils::createExchangerRef(pm);

  _exchangeData(exchanger.get());

  ItemInternalMap& edges_map = m_mesh->edgesMap();
  Integer nb_receiver = exchanger->nbReceiver();
  debug() << "NB RECEIVER=" << nb_receiver;
  Int64UniqueArray received_infos;
  for (Integer i = 0; i < nb_receiver; ++i) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    auto orig_rank = sm->destination();
    ISerializer* s = sm->serializer();
    s->setMode(ISerializer::ModeGet);
    Int64 nb_info = s->getInt64();
    if (is_verbose)
      info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
    received_infos.resize(nb_info);
    s->getSpan(received_infos);
    if ((nb_info % 3) != 0)
      ARCANE_FATAL("info size can not be divided by 3 x={0}", nb_info);
    Int64 nb_item = nb_info / 3;
    for (Int64 z = 0; z < nb_item; ++z) {
      Int64 old_uid = received_infos[(z * 3)];
      Int64 new_uid = received_infos[(z * 3) + 1];
      Int32 new_owner = static_cast<Int32>(received_infos[(z * 3) + 2]);
      //info() << "EDGE old_uid=" << old_uid << " new_uid=" << new_uid;
      impl::MutableItemBase iedge(edges_map.tryFind(old_uid));
      if (iedge.null())
        ARCANE_FATAL("Can not find own edge uid={0}", old_uid);
      iedge.setUniqueId(new_uid);
      iedge.setOwner(new_owner, m_my_rank);
      Edge edge{ iedge };
      if (is_verbose)
        info() << "SetEdgeOwner uid=" << new_uid << " owner=" << new_owner
               << " n0,n1=" << edge.node(0).uniqueId() << "," << edge.node(1).uniqueId()
               << " n0=" << ItemPrinter(edge.node(0)) << " n1=" << ItemPrinter(edge.node(1));
    }
  }
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
  cells_map.eachItem([&](Item cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    if (cell_uid>max_uid)
      max_uid = cell_uid;
  });
  info() << "Max uid=" << max_uid;
  Int32UniqueArray cell_first_edge_uid(max_uid+1);
  Int32UniqueArray cell_nb_num_back_edge(max_uid+1);
  Int32UniqueArray cell_true_boundary_edge(max_uid+1);

  cells_map.eachItem([&](Cell cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    Integer nb_num_back_edge = 0;
    Integer nb_true_boundary_edge = 0;
    for( Edge edge : cell.edges()){
      if (edge.itemBase().backCell()==cell)
        ++nb_num_back_edge;
      else if (edge.nbCell()==1){
        ++nb_true_boundary_edge;
      }
    }
    cell_nb_num_back_edge[cell_uid] = nb_num_back_edge;
    cell_true_boundary_edge[cell_uid] = nb_true_boundary_edge;
  });

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

  cells_map.eachItem([&](Cell cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    Integer nb_num_back_edge = 0;
    Integer nb_true_boundary_edge = 0;
    for( Edge edge : cell.edges() ){
      Int64 edge_new_uid = NULL_ITEM_UNIQUE_ID;
      //info() << "CHECK CELLUID=" << cell_uid << " EDGELID=" << edge->localId();
      if (edge.itemBase().backCell()==cell){
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
        edge.mutableItemBase().setUniqueId(edge_new_uid);
      }
    }
  });

  if (is_verbose){
    OStringStream ostr;
    cells_map.eachItem([&](Cell cell) {
      Int32 cell_uid = cell.uniqueId().asInt32();
      Integer index = 0;
      for( Edge edge : cell.edges() ){
        Int64 opposite_cell_uid = NULL_ITEM_UNIQUE_ID;
        bool true_boundary = false;
        bool internal_other = false;
        if (edge.itemBase().backCell()==cell){
        }
        else if (edge.nbCell()==1){
          true_boundary = true;
        }
        else{
          internal_other = true;
          opposite_cell_uid = edge.itemBase().backCell().uniqueId().asInt64();
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
    });
    info() << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeUniqueIdBuilder::
_computeEdgesUniqueIdsParallelV2()
{
  // Positionne les uniqueId() des arêtes de manière très simple.
  // Si le maximum des uniqueId() des noeuds est MAX_NODE_UID, alors
  // le uniqueId() d'une arête est :
  //
  // node(0).uniqueId() * MAX_NODE_UID + node(1).uniqueId()
  //
  // Cela ne fonctionne que si MAX_NODE_UID est inférieur à 2^31.

  IParallelMng* pm = m_mesh->parallelMng();

  ItemInternalMap& nodes_map = m_mesh->nodesMap();
  ItemInternalMap& edges_map = m_mesh->edgesMap();

  Int64 max_uid = 0;
  nodes_map.eachItem([&](Item node) {
    if (node.uniqueId() > max_uid)
      max_uid = node.uniqueId();
  });
  Int64 total_max_uid = pm->reduce(Parallel::ReduceMax,max_uid);
  if (total_max_uid>INT32_MAX)
    ARCANE_FATAL("Max uniqueId() for node is too big v={0} max_allowed={1}",total_max_uid,INT32_MAX);

  edges_map.eachItem([&](Edge edge) {
    Node node0{edge.node(0)};
    Node node1{edge.node(1)};
    Int64 new_uid = (node0.uniqueId().asInt64() * total_max_uid) + node1.uniqueId().asInt64();
    edge.mutableItemBase().setUniqueId(new_uid);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeUniqueIdBuilder::
_computeEdgesUniqueIdsParallel3()
{
  IParallelMng* pm = m_mesh->parallelMng();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  // Détermine le maximum des uniqueId() des noeuds
  Int64 my_max_node_uid = NULL_ITEM_UNIQUE_ID;
  nodes_map.eachItem([&](Item item) {
    Int64 node_uid = item.uniqueId();
    if (node_uid > my_max_node_uid)
      my_max_node_uid = node_uid;
  });
  Int64 global_max_node_uid = pm->reduce(Parallel::ReduceMax, my_max_node_uid);
  debug() << "NODE_UID_INFO: MY_MAX_UID=" << my_max_node_uid
          << " GLOBAL=" << global_max_node_uid;

  Parallel3EdgeUniqueIdBuilder builder(traceMng(), m_mesh_builder, global_max_node_uid);
  builder.compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EdgeUniqueIdBuilder::
_computeEdgesUniqueIdsParallel64bit()
{
  // Positionne les uniqueId() des arêtes
  // en utilisant un hash des deux nœuds de l'arête.
  ItemInternalMap& edges_map = m_mesh->edgesMap();

  std::hash<Int64> hasher;

  edges_map.eachItem([&](Edge edge) {
    Node node0{edge.node(0)};
    Node node1{edge.node(1)};
    size_t hash0 = hasher(node0.uniqueId().asInt64());
    size_t hash1 = hasher(node1.uniqueId().asInt64());
    hash0 ^= hash1 + 0x9e3779b9 + (hash0 << 6) + (hash0 >> 2);
    Int64 new_uid = hash0 & 0x7fffffff;
    edge.mutableItemBase().setUniqueId(new_uid);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
