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

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/HashTableMap2.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"

#include "arcane/parallel/BitonicSortT.H"

#include "arcane/mesh/ItemInternalMap.h"
#include "arcane/mesh/DynamicMesh.h"

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

  explicit ItemsOwnerBuilderImpl(DynamicMesh* mesh);

 public:

  void computeFacesOwner();

 private:

  DynamicMesh* m_mesh = nullptr;
  Int32 m_verbose_level = 0;
  UniqueArray<ItemOwnerInfo> m_items_owner_info;

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

  static bool compareLess(const ItemOwnerInfo& k1, const ItemOwnerInfo& k2)
  {
    if (k1.m_first_node_uid < k2.m_first_node_uid)
      return true;
    if (k1.m_first_node_uid > k2.m_first_node_uid)
      return false;

    if (k1.m_item_uid < k2.m_item_uid)
      return true;
    if (k1.m_item_uid > k2.m_item_uid)
      return false;

    if (k1.m_cell_uid < k2.m_cell_uid)
      return true;
    if (k1.m_cell_uid > k2.m_cell_uid)
      return false;

    if (k1.m_item_sender_rank < k2.m_item_sender_rank)
      return true;
    if (k1.m_item_sender_rank > k2.m_item_sender_rank)
      return false;

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsOwnerBuilderImpl::
ItemsOwnerBuilderImpl(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ITEMS_OWNER_BUILDER_IMPL_DEBUG_LEVEL", true))
    m_verbose_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsOwnerBuilderImpl::
computeFacesOwner()
{
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
/*!
 * \brief Tri les instances contenues dans m_items_owner_info replace les
 * valeurs triées dans ce même tableau.
 */
void ItemsOwnerBuilderImpl::
_sortInfos()
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 verbose_level = m_verbose_level;
  Parallel::BitonicSort<ItemOwnerInfo, ItemOwnerInfoSortTraits> items_sorter(pm);
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
  const Int32 nb_sorted = items_owner_info.size();
  info() << "NbSorted=" << nb_sorted;

  // Comme les informations d'une entité peuvent être réparties sur plusieurs rangs
  // après le tri, chaque rang envoie au rang suivant les informations
  // de la dernière entité de sa liste.

  UniqueArray<ItemOwnerInfo> items_owner_info_send_to_next;
  for (Int32 i = (nb_sorted - 1); i >= 0; --i) {
    const ItemOwnerInfo& x = items_owner_info[i];
    if (x.m_item_uid != items_owner_info[nb_sorted - 1].m_item_uid)
      break;
    items_owner_info_send_to_next.add(x);
  }
  Int32 nb_send_to_next = items_owner_info_send_to_next.size();
  info() << "NbSendToNext=" << nb_send_to_next;

  Int32 nb_to_receive_from_previous = 0;
  SmallArray<Parallel::Request> requests;
  // Envoie et recoit les tailles des tableaux
  if (my_rank != (nb_rank - 1))
    requests.add(pm->send(ConstArrayView<Int32>(1, &nb_send_to_next), my_rank + 1, false));
  if (my_rank > 0)
    requests.add(pm->recv(ArrayView<Int32>(1, &nb_to_receive_from_previous), my_rank - 1, false));

  pm->waitAllRequests(requests);
  requests.clear();

  // Envoie le tableau au suivant et récupère celui du précédent.
  UniqueArray<ItemOwnerInfo> items_owner_info_received_from_previous(nb_to_receive_from_previous);
  if (my_rank != (nb_rank - 1))
    requests.add(ItemOwnerInfoSortTraits::send(pm, my_rank + 1, items_owner_info_send_to_next));
  if (my_rank > 0)
    requests.add(ItemOwnerInfoSortTraits::recv(pm, my_rank - 1, items_owner_info_received_from_previous));
  pm->waitAllRequests(requests);

  const Int32 verbose_level = m_verbose_level;

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
  for (Int32 i = 0; i < nb_sorted; ++i) {
    const ItemOwnerInfo* first_ioi = &items_owner_info[i];
    Int64 item_uid = first_ioi->m_item_uid;
    // Si on est au début de la liste, prend l'entité envoyée par le rang précédent
    // si c'est la même que la nôtre.
    if (i == 0 && nb_to_receive_from_previous > 0) {
      ItemOwnerInfo* first_previous = &items_owner_info_received_from_previous[0];
      if (item_uid == first_previous->m_item_uid) {
        first_ioi = first_previous;
      }
    }
    // Si l'id courant est différent du précédent, on commence une nouvelle liste.
    if (item_uid != current_item_uid) {
      current_item_uid = item_uid;
      current_item_owner = first_ioi->m_cell_owner;
    }
    Int32 orig_sender = items_owner_info[i].m_item_sender_rank;
    UniqueArray<Int64>& send_array = resend_items_owner_info_map[orig_sender];
    send_array.add(current_item_uid);
    send_array.add(current_item_owner);
    if (verbose_level >= 2)
      info() << "SEND i=" << i << " rank=" << orig_sender << " item_uid=" << current_item_uid << " new_owner=" << current_item_owner;
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
        info() << "SetOwner uid=" << item_uid << " new_owner" << item_owner;
      x.toMutable().setOwner(item_owner, my_rank);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsOwnerBuilder::
ItemsOwnerBuilder(DynamicMesh* mesh)
: m_p(std::make_unique<ItemsOwnerBuilderImpl>(mesh))
{}

ItemsOwnerBuilder::
~ItemsOwnerBuilder()
{
  // Le destructeur doit être dans le '.cc' car 'ItemsOwnerBuilderImpl' n'est
  // pas connu dans le '.h'.
}

void ItemsOwnerBuilder::
computeFacesOwner()
{
  m_p->computeFacesOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
