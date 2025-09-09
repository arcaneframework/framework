// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsOwnerBuilder.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Classe pour calculer les propriétaires des entités.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemsOwnerBuilder.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/HashTableMap2.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"

#include "arcane/parallel/BitonicSortT.H"

#include "arcane/mesh/ItemInternalMap.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/DynamicMesh.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ce fichier contient un algorithme pour calculer les propriétaires des
 * entités autres que les mailles à partir des propriétaires des mailles.
 *
 * L'algorithme suppose que les propriétaires des mailles sont à jours et
 * synchronisés. Le propriétaire d'une entité sera alors le propriétaire de
 * la maille de plus petit uniqueId() connectée à cette entité.
 *
 * En parallèle, si une entité est à la frontière d'un sous-domaine, il n'est
 * pas possible de connaitre toutes les mailles qui y sont connectées.
 * Pour résoudre ce problème, on crée une liste des entités de la frontière
 * contenant pour chaque maille connectée un triplet (uniqueId() de l'entité,
 * uniqueId() de la maille connectée, owner() de la maille connectées).
 * Cette liste est ensuite triée en parallèle (via BitonicSort) par uniqueId()
 * de l'entité, puis par uniqueId() de la maille.
 * Pour déterminer le propriétaire d'une entité, il suffit ensuite de prendre
 * le propriétaire de la maille associée à la première occurrence de l'entité
 * dans cette liste triée. Une fois ceci fait, on renvoie aux rangs qui possèdent
 * cette entité cette information.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de l'algorithme de calcul des propriétaires.
 */
class ItemsOwnerBuilderImpl
: public TraceAccessor
{
  class ItemOwnerInfoSortTraits;

  /*!
   * \brief Informations sur une entité partagée.
   *
   * On conserve dans l'instance le uniqueId() du premier noeud
   * de l'entité et on s'en sert comme clé primaire pour le tri.
   * En général, les noeuds dont le uniqueId() est proche sont dans le même
   * sous-domaine. Comme on se sert de cette valeur comme clé primaire
   * du tri, cela permet de garantir une certaine cohérence topologique
   * des entités distribuées et ainsi éviter de faire un all-to-all qui
   * concerne un grand nombre de rangs.
   */
  class ItemOwnerInfo
  {
   public:

    ItemOwnerInfo() = default;
    ItemOwnerInfo(Int64 item_uid, Int64 first_node_uid, Int64 cell_uid, Int32 sender_rank, Int32 cell_owner)
    : m_item_uid(item_uid)
    , m_first_node_uid(first_node_uid)
    , m_cell_uid(cell_uid)
    , m_item_sender_rank(sender_rank)
    , m_cell_owner(cell_owner)
    {
    }

   public:

    //! uniqueId() de l'entité
    Int64 m_item_uid = NULL_ITEM_UNIQUE_ID;
    //! uniqueId() du premier noeud de l'entité
    Int64 m_first_node_uid = NULL_ITEM_UNIQUE_ID;
    //! uniqueId() de la maille à laquelle l'entité appartient
    Int64 m_cell_uid = NULL_ITEM_UNIQUE_ID;
    //! rang de celui qui a créé cette instance
    Int32 m_item_sender_rank = A_NULL_RANK;
    //! Propriétaire de la maille connectée à cette entité
    Int32 m_cell_owner = A_NULL_RANK;
  };

 public:

  explicit ItemsOwnerBuilderImpl(IMesh* mesh);

 public:

  void computeFacesOwner();
  void computeEdgesOwner();
  void computeNodesOwner();

 private:

  DynamicMesh* m_mesh = nullptr;
  Int32 m_verbose_level = 0;
  UniqueArray<ItemOwnerInfo> m_items_owner_info;
  /*!
   * \brief Indique comment effectuer le tri.
   *
   * Si vrai, on utilise la maille de plus petit uniqueId()
   * pour le tri. Sinon, c'est le plus petit rang. Cela servira
   * à déterminer qui sera le propriétaire d'une entité.
   */
  bool m_use_cell_uid_to_sort = true;

 private:

  void _sortInfos();
  void _processSortedInfos(ItemInternalMap& items_map);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemsOwnerBuilderImpl::ItemOwnerInfoSortTraits
{
 public:

  ItemOwnerInfoSortTraits(bool use_cell_uid_to_sort)
  : m_use_cell_uid_to_sort(use_cell_uid_to_sort)
  {}

 public:

  bool compareLess(const ItemOwnerInfo& k1, const ItemOwnerInfo& k2) const
  {
    if (k1.m_first_node_uid < k2.m_first_node_uid)
      return true;
    if (k1.m_first_node_uid > k2.m_first_node_uid)
      return false;

    if (k1.m_item_uid < k2.m_item_uid)
      return true;
    if (k1.m_item_uid > k2.m_item_uid)
      return false;

    if (m_use_cell_uid_to_sort) {
      if (k1.m_cell_uid < k2.m_cell_uid)
        return true;
      if (k1.m_cell_uid > k2.m_cell_uid)
        return false;

      if (k1.m_item_sender_rank < k2.m_item_sender_rank)
        return true;
      if (k1.m_item_sender_rank > k2.m_item_sender_rank)
        return false;
    }
    else {
      if (k1.m_item_sender_rank < k2.m_item_sender_rank)
        return true;
      if (k1.m_item_sender_rank > k2.m_item_sender_rank)
        return false;

      if (k1.m_cell_uid < k2.m_cell_uid)
        return true;
      if (k1.m_cell_uid > k2.m_cell_uid)
        return false;
    }
    // ke.node2_uid == k2.node2_uid
    return (k1.m_cell_owner < k2.m_cell_owner);
  }

  static Parallel::Request send(IParallelMng* pm, Int32 rank, ConstArrayView<ItemOwnerInfo> values)
  {
    const ItemOwnerInfo* fsi_base = values.data();
    return pm->send(ByteConstArrayView(messageSize(values), reinterpret_cast<const Byte*>(fsi_base)), rank, false);
  }
  static Parallel::Request recv(IParallelMng* pm, Int32 rank, ArrayView<ItemOwnerInfo> values)
  {
    ItemOwnerInfo* fsi_base = values.data();
    return pm->recv(ByteArrayView(messageSize(values), reinterpret_cast<Byte*>(fsi_base)), rank, false);
  }
  static Integer messageSize(ConstArrayView<ItemOwnerInfo> values)
  {
    return CheckedConvert::toInteger(values.size() * sizeof(ItemOwnerInfo));
  }
  static ItemOwnerInfo maxValue()
  {
    return ItemOwnerInfo(INT64_MAX, INT64_MAX, INT64_MAX, INT32_MAX, INT32_MAX);
  }
  static bool isValid(const ItemOwnerInfo& fsi)
  {
    return fsi.m_item_uid != INT64_MAX;
  }

 private:

  bool m_use_cell_uid_to_sort = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsOwnerBuilderImpl::
ItemsOwnerBuilderImpl(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
{
  auto* dm = dynamic_cast<DynamicMesh*>(mesh);
  if (!dm)
    ARCANE_FATAL("Mesh is not an instance of 'DynamicMesh'");
  m_mesh = dm;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ITEMS_OWNER_BUILDER_IMPL_DEBUG_LEVEL", true))
    m_verbose_level = v.value();
  // Indique si on trie en fonction de la maille de plus petit uniqueId()
  // ou en fonction du plus petit rang.
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ITEMS_OWNER_BUILDER_USE_RANK", true))
    m_use_cell_uid_to_sort = !v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilderImpl::
computeFacesOwner()
{
  m_items_owner_info.clear();

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();
  FaceFamily& face_family = m_mesh->trueFaceFamily();

  info() << "** BEGIN ComputeFacesOwner nb_face=" << faces_map.count();

  // Parcours toutes les faces.
  // Ne garde que celles qui sont frontières ou dont les propriétaires des
  // deux mailles de part et d'autre sont différents de notre sous-domaine.
  UniqueArray<Int32> faces_to_add;
  UniqueArray<Int64> faces_to_add_uid;
  faces_map.eachItem([&](Face face) {
    Int32 nb_cell = face.nbCell();
    if (nb_cell == 2)
      if (face.cell(0).owner() == my_rank && face.cell(1).owner() == my_rank) {
        face.mutableItemBase().setOwner(my_rank, my_rank);
        return;
      }
    faces_to_add.add(face.localId());
    faces_to_add_uid.add(face.uniqueId());
  });
  info() << "ItemsOwnerBuilder: NB_FACE_TO_TRANSFER=" << faces_to_add.size();
  const Int32 verbose_level = m_verbose_level;

  FaceInfoListView faces(&face_family);
  for (Int32 lid : faces_to_add) {
    Face face(faces[lid]);
    Int64 face_uid = face.uniqueId();
    for (Cell cell : face.cells()) {
      if (verbose_level >= 2)
        info() << "ADD lid=" << lid << " uid=" << face_uid << " cell_uid=" << cell.uniqueId() << " owner=" << cell.owner();
      m_items_owner_info.add(ItemOwnerInfo(face_uid, face.node(0).uniqueId(), cell.uniqueId(), my_rank, cell.owner()));
    }
  }

  // Tri les instances de ItemOwnerInfo et les place les valeurs triées
  // dans items_owner_info.
  _sortInfos();
  _processSortedInfos(faces_map);

  face_family.notifyItemsOwnerChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilderImpl::
computeEdgesOwner()
{
  m_items_owner_info.clear();

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  ItemInternalMap& edges_map = m_mesh->edgesMap();
  EdgeFamily& edge_family = m_mesh->trueEdgeFamily();

  info() << "** BEGIN ComputeEdgesOwner nb_edge=" << edges_map.count();

  // Parcours toutes les arêtes.
  // Ne garde que celles qui sont frontières ou dont au moins un des propriétaires des
  // mailles connectées sont différents de notre sous-domaine.
  UniqueArray<Int32> edges_to_add;
  UniqueArray<Int64> edges_to_add_uid;
  // La force brute ajoute toutes les arêtes.
  // Cela permet de garantir qu'on calcule bien les propriétaires, même si
  // on ne sait pas déterminer les arêtes frontìères.
  // Cela n'est pas optimum, car on envoie aussi nos arêtes internes
  // alors qu'on est certain qu'on est leur propriétaire.
  bool do_brute_force = true;
  edges_map.eachItem([&](Edge edge) {
    Int32 nb_cell = edge.nbCell();
    Int32 nb_cell_with_my_rank = 0;
    for (Cell cell : edge.cells())
      if (cell.owner() == my_rank)
        ++nb_cell_with_my_rank;
    bool do_add = (nb_cell == 1 || (nb_cell != nb_cell_with_my_rank));
    if (do_add || do_brute_force) {
      edges_to_add.add(edge.localId());
      edges_to_add_uid.add(edge.uniqueId());
    }
    else
      edge.mutableItemBase().setOwner(my_rank, my_rank);
  });
  info() << "ItemsOwnerBuilder: NB_FACE_TO_TRANSFER=" << edges_to_add.size();
  const Int32 verbose_level = m_verbose_level;

  EdgeInfoListView edges(&edge_family);
  for (Int32 lid : edges_to_add) {
    Edge edge(edges[lid]);
    Int64 edge_uid = edge.uniqueId();
    for (Cell cell : edge.cells()) {
      if (verbose_level >= 2)
        info() << "ADD lid=" << lid << " uid=" << edge_uid << " cell_uid=" << cell.uniqueId() << " owner=" << cell.owner();
      m_items_owner_info.add(ItemOwnerInfo(edge_uid, edge.node(0).uniqueId(), cell.uniqueId(), my_rank, cell.owner()));
    }
  }

  // Tri les instances de ItemOwnerInfo et les place les valeurs triées
  // dans items_owner_info.
  _sortInfos();
  _processSortedInfos(edges_map);

  edge_family.notifyItemsOwnerChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilderImpl::
computeNodesOwner()
{
  m_items_owner_info.clear();

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();
  NodeFamily& node_family = m_mesh->trueNodeFamily();

  info() << "** BEGIN ComputeNodesOwner nb_node=" << nodes_map.count();

  // Place par défaut tous les noeuds dans ce sous-domaine
  nodes_map.eachItem([&](Node node) {
    node.mutableItemBase().setOwner(my_rank, my_rank);
  });

  // Parcours toutes les faces.
  // Ne garde que celles qui sont frontières ou dont les propriétaires des
  // deux mailles de part et d'autre sont différents de notre sous-domaine.
  //UniqueArray<Int32> faces_to_add;
  //faces_map.eachItem([&](Face face) {
  //Int32 nb_cell = face.nbCell();
  // if (nb_cell == 1)
  //  faces_to_add.add(face.localId());
  //});
  //info() << "ItemsOwnerBuilder: NB_FACE_TO_ADD=" << faces_to_add.size();
  const Int32 verbose_level = m_verbose_level;

  // Ajoute tous les noeuds des faces frontières.
  std::unordered_set<Int32> done_nodes;

  FaceInfoListView faces(m_mesh->faceFamily());
  UniqueArray<Int32> nodes_to_add;
  faces_map.eachItem([&](Face face) {
    Int32 face_nb_cell = face.nbCell();
    if (face_nb_cell == 2)
      return;
    for (Node node : face.nodes()) {
      Int32 node_id = node.localId();
      if (done_nodes.find(node_id) == done_nodes.end()) {
        nodes_to_add.add(node_id);
        done_nodes.insert(node_id);
        node.mutableItemBase().setOwner(A_NULL_RANK, my_rank);
      }
    }
  });

  info() << "ItemsOwnerBuilder: NB_NODE_TO_ADD=" << nodes_to_add.size();
  NodeInfoListView nodes(&node_family);
  for (Int32 lid : nodes_to_add) {
    Node node(nodes[lid]);
    Int64 node_uid = node.uniqueId();
    for (Cell cell : node.cells()) {
      if (verbose_level >= 2)
        info() << "ADD lid=" << lid << " uid=" << node_uid << " cell_uid=" << cell.uniqueId() << " owner=" << cell.owner();
      m_items_owner_info.add(ItemOwnerInfo(node_uid, node_uid, cell.uniqueId(), my_rank, cell.owner()));
    }
  }

  // Tri les instances de ItemOwnerInfo et les place les valeurs triées
  // dans items_owner_info.
  _sortInfos();
  _processSortedInfos(nodes_map);

  node_family.notifyItemsOwnerChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tri les instances contenues dans m_items_owner_info replace les
 * valeurs triées dans ce même tableau.
 */
void ItemsOwnerBuilderImpl::
_sortInfos()
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 verbose_level = m_verbose_level;
  ItemOwnerInfoSortTraits sort_traits(m_use_cell_uid_to_sort);
  Parallel::BitonicSort<ItemOwnerInfo, ItemOwnerInfoSortTraits> items_sorter(pm, sort_traits);
  items_sorter.setNeedIndexAndRank(false);
  Real sort_begin_time = platform::getRealTime();
  items_sorter.sort(m_items_owner_info);
  Real sort_end_time = platform::getRealTime();
  m_items_owner_info = items_sorter.keys();
  info() << "END_ALL_ITEM_OWNER_SORTER time=" << (Real)(sort_end_time - sort_begin_time);
  if (verbose_level >= 2)
    for (const ItemOwnerInfo& x : m_items_owner_info) {
      info() << "Sorted first_node_uid=" << x.m_first_node_uid << " item_uid="
             << x.m_item_uid << " cell_uid=" << x.m_cell_uid << " owner=" << x.m_cell_owner;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilderImpl::
_processSortedInfos(ItemInternalMap& items_map)
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  const Int32 nb_rank = pm->commSize();
  ConstArrayView<ItemOwnerInfo> items_owner_info = m_items_owner_info;
  Int32 nb_sorted = items_owner_info.size();
  info() << "NbSorted=" << nb_sorted;
  const bool is_last_rank = ((my_rank + 1) == nb_rank);
  const bool is_first_rank = (my_rank == 0);
  const Int32 verbose_level = m_verbose_level;

  // Comme les informations d'une entité peuvent être réparties sur plusieurs rangs
  // après le tri, chaque rang envoie au rang suivant les informations
  // de la dernière entité de sa liste.
  // Il faut faire attention à bien envoyer la liste dans le même ordre trié pour
  // garantir la cohérence car on va prendre le premier élément de la liste pour
  // positionner le propriétaire.

  UniqueArray<ItemOwnerInfo> items_owner_info_send_to_next;
  if (nb_sorted > 0 && !is_last_rank) {
    Int32 send_index = nb_sorted;
    Int64 last_uid = items_owner_info[nb_sorted - 1].m_item_uid;
    for (Int32 i = (nb_sorted - 1); i >= 0; --i) {
      const ItemOwnerInfo& x = items_owner_info[i];
      if (x.m_item_uid != last_uid) {
        send_index = i + 1;
        break;
      }
    }
    info() << "SendIndext=" << send_index << " nb_sorted=" << nb_sorted;
    for (Int32 i = send_index; i < nb_sorted; ++i) {
      const ItemOwnerInfo& x = items_owner_info[i];
      items_owner_info_send_to_next.add(x);
      if (verbose_level >= 2)
        info() << "AddSendToNext item_uid=" << x.m_item_uid << " owner=" << x.m_cell_owner
               << " from_rank=" << x.m_item_sender_rank << " index=" << i;
    }
  }
  Int32 nb_send_to_next = items_owner_info_send_to_next.size();
  info() << "NbSendToNext=" << nb_send_to_next;

  Int32 nb_to_receive_from_previous = 0;
  SmallArray<Parallel::Request> requests;
  // Envoie et recoit les tailles des tableaux
  if (!is_last_rank)
    requests.add(pm->send(ConstArrayView<Int32>(1, &nb_send_to_next), my_rank + 1, false));
  if (!is_first_rank)
    requests.add(pm->recv(ArrayView<Int32>(1, &nb_to_receive_from_previous), my_rank - 1, false));

  pm->waitAllRequests(requests);
  requests.clear();

  // Envoie le tableau au suivant et récupère celui du précédent.
  UniqueArray<ItemOwnerInfo> items_owner_info_received_from_previous(nb_to_receive_from_previous);
  if (!is_last_rank)
    requests.add(ItemOwnerInfoSortTraits::send(pm, my_rank + 1, items_owner_info_send_to_next));
  if (!is_first_rank)
    requests.add(ItemOwnerInfoSortTraits::recv(pm, my_rank - 1, items_owner_info_received_from_previous));
  pm->waitAllRequests(requests);

  // Supprime de la liste les entités qu'on a envoyé au suivant
  m_items_owner_info.resize(nb_sorted - nb_send_to_next);
  items_owner_info = m_items_owner_info.view();
  nb_sorted = items_owner_info.size();
  info() << "NbRemaining=" << nb_sorted;

  Int64 current_item_uid = NULL_ITEM_UNIQUE_ID;
  Int32 current_item_owner = A_NULL_RANK;

  // Parcours la liste des entités qu'on a recu.
  // Chaque entité est présente plusieurs fois dans la liste : au moins
  // une fois par maille connectée à cette entité. Comme cette liste est triée
  // par uniqueId() croissant de ces mailles, et que c'est la maille de plus
  // petit uniqueId() qui donne le propriétaire de l'entité, alors le propriétaire
  // est celui du premier élément de cette liste.
  // On envoie ensuite à tous les rangs qui possèdent cette entité ce nouveau propriétaire.
  // Le tableau envoyé contient une liste de couples (item_uid, item_new_owner).
  impl::HashTableMap2<Int32, UniqueArray<Int64>> resend_items_owner_info_map;
  for (Int32 index = 0; index < (nb_sorted + nb_to_receive_from_previous); ++index) {
    const ItemOwnerInfo* first_ioi = nullptr;
    // Si \a i est inférieur à nb_to_receive_from_previous, on prend
    // les informations de la liste recue.
    if (index < nb_to_receive_from_previous)
      first_ioi = &items_owner_info_received_from_previous[index];
    else
      first_ioi = &items_owner_info[index - nb_to_receive_from_previous];
    Int64 item_uid = first_ioi->m_item_uid;

    // Si l'id courant est différent du précédent, on commence une nouvelle liste.
    if (item_uid != current_item_uid) {
      current_item_uid = item_uid;
      if (m_use_cell_uid_to_sort)
        current_item_owner = first_ioi->m_cell_owner;
      else
        current_item_owner = first_ioi->m_item_sender_rank;
      if (verbose_level >= 2)
        info() << "SetOwner from sorted index=" << index << " item_uid=" << current_item_uid << " new_owner=" << current_item_owner;
    }
    Int32 orig_sender = first_ioi->m_item_sender_rank;
    UniqueArray<Int64>& send_array = resend_items_owner_info_map[orig_sender];
    send_array.add(current_item_uid);
    send_array.add(current_item_owner);
    if (verbose_level >= 2)
      info() << "SEND i=" << index << " rank=" << orig_sender << " item_uid=" << current_item_uid << " new_owner=" << current_item_owner;
  }

  auto exchanger{ ParallelMngUtils::createExchangerRef(pm) };
  info() << "NbResendRanks=" << resend_items_owner_info_map.size();
  for (const auto& [key, value] : resend_items_owner_info_map) {
    if (verbose_level >= 1)
      info() << "RESEND_INFO to_rank=" << key << " nb=" << value.size();
    exchanger->addSender(key);
  }
  exchanger->initializeCommunicationsMessages();
  {
    Int32 index = 0;
    for (const auto& [key, value] : resend_items_owner_info_map) {
      ISerializeMessage* sm = exchanger->messageToSend(index);
      ++index;
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeReserve);
      s->reserveArray(value);
      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);
      s->putArray(value);
    }
  }
  exchanger->processExchange();
  UniqueArray<Int64> receive_info;

  for (Integer i = 0, ns = exchanger->nbReceiver(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    s->setMode(ISerializer::ModeGet);
    s->getArray(receive_info);
    Int32 receive_size = receive_info.size();
    if (verbose_level >= 1)
      info() << "RECEIVE_INFO size=" << receive_size << " rank2=" << sm->destination();
    // Vérifie que la taille est un multiple de 2
    if ((receive_size % 2) != 0)
      ARCANE_FATAL("Size '{0}' is not a multiple of 2", receive_size);
    Int32 buf_size = receive_size / 2;
    for (Int32 z = 0; z < buf_size; ++z) {
      Int64 item_uid = receive_info[z * 2];
      Int32 item_owner = CheckedConvert::toInt32(receive_info[(z * 2) + 1]);
      impl::ItemBase x = items_map.findItem(item_uid);
      if (verbose_level >= 2)
        info() << "SetOwner uid=" << item_uid << " new_owner=" << item_owner;
      x.toMutable().setOwner(item_owner, my_rank);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsOwnerBuilder::
ItemsOwnerBuilder(IMesh* mesh)
: m_p(std::make_unique<ItemsOwnerBuilderImpl>(mesh))
{}

ItemsOwnerBuilder::
~ItemsOwnerBuilder()
{
  // Le destructeur doit être dans le '.cc' car 'ItemsOwnerBuilderImpl' n'est
  // pas connu dans le '.h'.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilder::
computeFacesOwner()
{
  m_p->computeFacesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilder::
computeEdgesOwner()
{
  m_p->computeEdgesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilder::
computeNodesOwner()
{
  m_p->computeNodesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
