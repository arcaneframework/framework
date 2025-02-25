// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataWriter.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Lecteur/Ecrivain de IData en parallèle.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/ParallelDataWriter.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/Math.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IData.h"
#include "arcane/core/parallel/BitonicSortT.H"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelDataWriter::Impl
: public TraceAccessor
{
 public:

  explicit Impl(IParallelMng* pm);

 public:

  Int64ConstArrayView sortedUniqueIds() const;
  void setGatherAll(bool v);

 private:

  IParallelMng* m_parallel_mng = nullptr;
  //! Tableau indiquant les rangs des process dont on recoit des infos
  UniqueArray<Int32> m_ranks_to_send;
  //! Tableau indiquant les rangs des process auxquels on envoie des infos
  UniqueArray<Int32> m_ranks_to_recv;
  //TODO ne pas utiliser un tableau dimensionné au commSize()
  UniqueArray<UniqueArray<Int32>> m_indexes_to_send;
  UniqueArray<UniqueArray<Int32>> m_indexes_to_recv;
  Int32 m_nb_item = 0;
  Int64UniqueArray m_sorted_unique_ids;

  UniqueArray<Int32> m_local_indexes_to_send;
  UniqueArray<Int32> m_local_indexes_to_recv;

  bool m_gather_all = false;
  bool m_is_verbose = false;

 public:

  void sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid);

  Ref<IData> getSortedValues(IData* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataWriter::
ParallelDataWriter(IParallelMng* pm)
: m_p(new Impl(pm))
{
}

ParallelDataWriter::
~ParallelDataWriter()
{
  delete m_p;
}

Int64ConstArrayView ParallelDataWriter::
sortedUniqueIds() const
{
  return m_p->sortedUniqueIds();
}
void ParallelDataWriter::
setGatherAll(bool v)
{
  m_p->setGatherAll(v);
}

void ParallelDataWriter::
sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid)
{
  m_p->sort(local_ids,items_uid);
}

Ref<IData> ParallelDataWriter::
getSortedValues(IData* data)
{
  return m_p->getSortedValues(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataWriter::Impl::
Impl(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView ParallelDataWriter::Impl::
sortedUniqueIds() const
{
  return m_sorted_unique_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataWriter::Impl::
setGatherAll(bool v)
{
  m_gather_all = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataWriter::Impl::
sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid)
{
  IParallelMng* pm = m_parallel_mng;

  Parallel::BitonicSort<Int64> uid_sorter(pm);
  uid_sorter.sort(items_uid);

  ConstArrayView<Int32> key_indexes = uid_sorter.keyIndexes();
  ConstArrayView<Int32> key_ranks = uid_sorter.keyRanks();
  ConstArrayView<Int64> keys = uid_sorter.keys();

  UniqueArray<Int64> global_all_keys;
  UniqueArray<Int32> global_all_key_indexes;
  UniqueArray<Int32> global_all_key_ranks;

  Int32 nb_item = keys.size();
  const Int32 my_rank = pm->commRank();
  const bool is_verbose = m_is_verbose;
  if (is_verbose) {
    for (Integer i = 0; i < math::min(nb_item, 20); ++i) {
      info() << "ORIGINAL I=" << i << " UID=" << items_uid[i]
             << " INDEX=" << key_indexes[i]
             << " RANK=" << key_ranks[i]
             << " KEY=" << keys[i];
    }
  }

  if (m_gather_all){
    // Le proc 0 récupère tout
    pm->allGatherVariable(keys,global_all_keys);
    pm->allGatherVariable(key_indexes,global_all_key_indexes);
    pm->allGatherVariable(key_ranks,global_all_key_ranks);
    Int32 gather_rank = 0;

    if (pm->commRank()!=gather_rank){
      global_all_key_ranks.clear();
      global_all_key_indexes.clear();
      global_all_keys.clear();
    }
    nb_item = global_all_keys.size();
      
    key_ranks = global_all_key_ranks.view();
    key_indexes = global_all_key_indexes.view();
    keys = global_all_keys.view();
  }

  m_nb_item = nb_item;

  m_sorted_unique_ids.resize(nb_item);
  m_sorted_unique_ids.copy(keys);

  //info() << "END SORT SIZE=" << nb_item << " KEY_SIZE=" << keys.size();
  if (is_verbose) {
    for (Integer i = 0; i < math::min(nb_item, 20); ++i) {
      info() << "I=" << i << " KEY=" << keys[i]
             << " INDEX=" << key_indexes[i]
             << " RANK=" << key_ranks[i];
    }
  }
  {
    UniqueArray<UniqueArray<Int32>> indexes_list(pm->commSize());
    UniqueArray<UniqueArray<Int32>> own_indexes_list(pm->commSize());
    auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };

    for( Integer i=0; i<nb_item; ++i ){
      Int32 index = key_indexes[i];
      Int32 rank = key_ranks[i];
      if (rank != my_rank && indexes_list[rank].empty())
        sd_exchange->addSender(rank);
      indexes_list[rank].add(index);
      own_indexes_list[rank].add(i);
    }
    m_local_indexes_to_recv = own_indexes_list[my_rank];
    m_local_indexes_to_send.resize(indexes_list[my_rank].size());

    sd_exchange->initializeCommunicationsMessages();
    //info() << "NB SEND=" << sd_exchange->nbSender()
    //       << " NB_RECV=" << sd_exchange->nbReceiver();
    Int32ConstArrayView send_sd = sd_exchange->senderRanks();
    Integer nb_send = send_sd.size();
    m_indexes_to_recv.resize(nb_send);
    m_ranks_to_recv.resize(nb_send);
    for( Integer i=0; i<nb_send; ++i ){
      //info() << " SEND TO A: rank=" << send_sd[i];
      ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
      Int32 dest_rank = send_sd[i];
      ISerializer* serializer = send_msg->serializer();
      m_indexes_to_recv[i] = own_indexes_list[dest_rank];
      m_ranks_to_recv[i] = dest_rank;
      serializer->setMode(ISerializer::ModeReserve);

      serializer->reserveArray(indexes_list[dest_rank]);

      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);

      serializer->putArray(indexes_list[dest_rank]);
      if (is_verbose) {
        Integer nb_to_send = indexes_list[dest_rank].size();
        for (Integer z = 0; z < nb_to_send; ++z) {
          Integer index = indexes_list[dest_rank][z];
          info() << " SEND Z=" << z << " RANK=" << dest_rank << " index=" << index;
        }
      }
    }
    sd_exchange->processExchange();

    ConstArrayView<Int32> recv_sd = sd_exchange->receiverRanks();
    const Int32 nb_recv = recv_sd.size();
    m_indexes_to_send.resize(nb_recv);
    m_ranks_to_send.resize(nb_recv);
    for( Integer i=0; i<nb_recv; ++i ){
      //info() << " RECEIVE FROM A: rank=" << recv_sd[i];
      ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
      Int32 orig_rank = recv_sd[i];
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      Int32Array& recv_indexes = m_indexes_to_send[i];
      m_ranks_to_send[i] = orig_rank;
      serializer->getArray(recv_indexes);
      const Int32 nb_to_recv = recv_indexes.size();
      //info() << " RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv << " S2=" << own_group_local_ids.size();
      for( Integer z=0; z<nb_to_recv; ++z ){
        Int32 index = recv_indexes[z];
        //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
        //     << " index2=" << own_group_local_ids[index];
        recv_indexes[z] = local_ids[index];
      }
      //info() << "END RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv;
    }

    // Traite les entités locales
    {
      const Int32 nb_local = m_local_indexes_to_send.size();
      for (Int32 z = 0; z < nb_local; ++z) {
        Int32 index = indexes_list[my_rank][z];
        //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
        //     << " index2=" << own_group_local_ids[index];
        m_local_indexes_to_send[z] = local_ids[index];
      }
    }

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> ParallelDataWriter::Impl::
getSortedValues(IData* data)
{
  IParallelMng* pm = m_parallel_mng;
  Ref<IData> sorted_data = data->cloneEmptyRef();

  auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };
  for (Int32 rank_to_send : m_ranks_to_send)
    sd_exchange->addSender(rank_to_send);

  UniqueArray<Int32> ranks_to_recv2;
  for (Int32 rank_to_receive : m_ranks_to_recv)
    ranks_to_recv2.add(rank_to_receive);

  sd_exchange->initializeCommunicationsMessages(ranks_to_recv2);
  Int32ConstArrayView send_sd = sd_exchange->senderRanks();
  const Int32 nb_send = send_sd.size();
  for( Integer i=0; i<nb_send; ++i ){
    //info() << " SEND TO B: rank=" << send_sd[i];
    ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
    //Int32 dest_rank = send_sd[i];
    ISerializer* serializer = send_msg->serializer();
    serializer->setMode(ISerializer::ModeReserve);
    data->serialize(serializer,m_indexes_to_send[i],0);
    serializer->allocateBuffer();
    serializer->setMode(ISerializer::ModePut);
    data->serialize(serializer,m_indexes_to_send[i],0);
  }

  sd_exchange->processExchange();

  Int32ConstArrayView recv_sd = sd_exchange->receiverRanks();
  const Int32 nb_recv = recv_sd.size();
  sorted_data->resize(m_nb_item);
  for( Integer i=0; i<nb_recv; ++i ){
    //info() << " RECEIVE FROM B: rank=" << recv_sd[i];
    ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
    //Int32 orig_rank = recv_sd[i];
    ISerializer* serializer = recv_msg->serializer();
    serializer->setMode(ISerializer::ModeGet);
    sorted_data->serialize(serializer,m_indexes_to_recv[i],0);
  }

  // Traite les données qui sont déjà présentes sur ce processeur.
  {
    ConstArrayView<Int32> local_recv_indexes = m_local_indexes_to_recv;
    const Int32 nb_local_index = local_recv_indexes.size();
    if (nb_local_index>0){
      SerializeBuffer sbuf;
      sbuf.setMode(ISerializer::ModeReserve);
      data->serialize(&sbuf, m_local_indexes_to_send, nullptr);
      sbuf.allocateBuffer();
      sbuf.setMode(ISerializer::ModePut);
      data->serialize(&sbuf, m_local_indexes_to_send, nullptr);
      sbuf.setMode(ISerializer::ModeGet);
      sorted_data->serialize(&sbuf, local_recv_indexes, nullptr);
    }
  }
  return sorted_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ParallelDataWriter> ParallelDataWriterList::
getOrCreateWriter(const ItemGroup& group)
{
  auto i = m_data_writers.find(group);
  if (i != m_data_writers.end())
    return i->second;
  IParallelMng* pm = group.itemFamily()->parallelMng();
  Ref<ParallelDataWriter> writer = makeRef(new ParallelDataWriter(pm));
  {
    Int64UniqueArray items_uid;
    ItemGroup own_group = group.own();
    MeshUtils::fillUniqueIds(own_group.view(), items_uid);
    Int32ConstArrayView local_ids = own_group.internal()->itemsLocalId();
    writer->sort(local_ids, items_uid);
  }
  m_data_writers.try_emplace(group, writer);
  return writer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
